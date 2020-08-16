import asyncio
from collections import deque

import websocket
import threading, queue
import json
import time
import numpy as np
import cv2


class GamesManager:
    class MessageType:
        NEW_GAME_REQUEST = 1
        REMOVE_GAME = 2
        NEW_GAME_ACCEPTED = 3

    class MessageClientSenderStub:
        @staticmethod
        def new_game_request(seed, n_pallets):
            return {"message_intention": GamesManager.MessageType.NEW_GAME_REQUEST, "seed": seed,
                    "n_pallets": n_pallets}

        @staticmethod
        def remove_game(port):
            return {"message_intention": GamesManager.MessageType.REMOVE_GAME, "port": int(port)}

    def __init__(self, state_size=800, n_frames_state=1, n_pallets=20, max_steps=500):
        self.host = "localhost"
        self.port = 9000
        self.games = dict()
        self.anonymous_games = dict()
        self.state_size = state_size
        self.n_frames_state = n_frames_state
        self.n_pallets = n_pallets
        self.max_steps = max_steps

        self.listen_thread = threading.Thread(target=self.run)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        self.client = Client(host=self.host, port=self.port)
        self.client.start()
        conn_timeout = 100
        while not self.is_ready() and conn_timeout:
            time.sleep(0.05)
            conn_timeout -= 1
        self.game_count = 0
        self.games_requested_pending = 0
        self.games_requested_pending_lock = threading.Lock()

    def get_n_games(self):
        return len(self.games)

    def get_game(self, identifier):
        if identifier in self.games:
            return self.games[identifier]
        elif identifier in self.anonymous_games:
            return self.anonymous_games[identifier]

    def is_ready(self):
        return self.client.ws.sock is not None and self.client.ws.sock.connected is True

    def run(self):
        pass

    def remove_game(self, port):
        # print("Remove request sent.")
        msg_out = GamesManager.MessageClientSenderStub.remove_game(port)
        msg_out = json.dumps(msg_out)
        self.client.send(msg_out)

        if port in self.games:
            game = self.games.pop(port)  # remove from container
        elif port in self.anonymous_games:
            game = self.anonymous_games.pop(port)

        del game

    def request_new_game(self, name, pooled=True):
        self.game_count += 1
        msg_out = GamesManager.MessageClientSenderStub.new_game_request(seed=self.game_count, n_pallets=self.n_pallets)
        msg_out = json.dumps(msg_out)
        self.client.send(msg_out)
        # print("Request sent.")
        # print("Waiting for acceptance.")
        try:
            msg_in = self.client.receive(block=True, timeout=10) # waits 10 sec
        except queue.Empty:
            print("Time out, retry ...")
            return self.request_new_game(name, pooled)


        msg_in = json.loads(msg_in)

        if msg_in["message_intention"] == GamesManager.MessageType.NEW_GAME_ACCEPTED:
            # print("Request accepted.")
            accepted_port = msg_in["port"]
            game = Game(port=accepted_port, name=name, state_size=self.state_size, n_frames_state=self.n_frames_state,
                        max_steps=self.max_steps)

            if pooled is True:
                self.games[accepted_port] = game
            else:
                self.anonymous_games[accepted_port] = game

            with self.games_requested_pending_lock:
                if self.games_requested_pending is not None:
                    self.games_requested_pending -= 1
                if self.games_requested_objects is not None:
                    self.games_requested_objects[game.get_identifier()] = game
        else:
            return None

        return game

    def request_new_games(self, n_games, name, pooled=True):
        self.games_requested_pending = n_games
        self.games_requested_objects = {}
        for i in range(n_games):
            t = threading.Thread(target=self.request_new_game, args=(name, pooled,))
            t.daemon = True
            t.start()

        while self.games_requested_pending > 0:
            print("Game request pending: {}".format(self.games_requested_pending), end="\n")
            time.sleep(0.25)

        return self.games_requested_objects

    def get_current_states(self):
        ret = []
        for port in self.games:
            game = self.games[port]
            ret.append(game.get_current_state())

        return ret

    def act_and_observe(self, actions, replace_terminal_game=True, max_steps=None):
        tuples = []
        pending_requests_lock = threading.Lock()

        pending_requests_count = [0 for _ in range(len(actions.keys()))]
        windows_to_close = []  # windows have to be closed from main thread for some reason, this is the buffer
        if isinstance(actions, dict):

            def worker(port):
                action = actions[port]

                if port in self.games:
                    game = self.games[port]
                else:
                    game = self.anonymous_games[port]

                tuple = game.act_and_observe(action, max_steps=max_steps)

                with pending_requests_lock:
                    tuples.append(tuple)

                is_animated = game.animated

                if "is_terminal" in tuple and tuple["is_terminal"] == 1.0:
                    name = game.name
                    self.remove_game(port)

                    if replace_terminal_game is True:
                        self.request_new_game(name)

                with pending_requests_lock:
                    pending_requests_count.pop(0)
                    if is_animated and "is_terminal" in tuple and tuple["is_terminal"] == 1.0:
                        windows_to_close.append(port)

            for port in actions:
                t = threading.Thread(target=worker, args=(port,))
                t.daemon = True
                t.start()

            received_all = False
            while received_all is False:
                with pending_requests_lock:
                    if len(pending_requests_count) == 0:
                        received_all = True
                if received_all == False:
                    time.sleep(0.01)

            for port in windows_to_close:
                cv2.waitKey(1)
                cv2.destroyWindow(str(port))
                cv2.waitKey(1)

        elif isinstance(actions, list):
            for i, port in enumerate(self.games):
                game = self.games[port]
                action = actions[i]
                tuple = game.act_and_observe(action)

                if "is_terminal" in tuple and tuple["is_terminal"] == 1.0:
                    name = self.games[port].name
                    self.remove_game(port)
                    self.request_new_game(name)

                tuples.append(tuple)

        return tuples

    def get_states_rewards(self):
        ret = []
        for game in self.games.values():
            state_reward = game.get_state_reward()
            if state_reward is not None:
                ret.append(state_reward)

        return ret


class Game:
    def __init__(self, port, name, state_size=800, n_frames_state=1, max_steps=500):
        self.name = name
        self.port = port
        self.state_size = state_size
        self.n_frames_state = n_frames_state
        self.max_steps = max_steps
        self.host = "localhost"
        self.client = Client(host=self.host, port=port)  # (host="echo.websocket.org", port="80")
        self.client.start()

        self.life = 0
        self.prev_pellets = None

        meshgrid = np.meshgrid(range(self.state_size), range(self.state_size), indexing="ij")
        self.coordinates = np.concatenate([meshgrid[0].flatten().reshape((-1, 1)), meshgrid[1].flatten().reshape((-1, 1))], axis=1)
        center = np.array([(state_size-1)/2, (state_size-1)/2]) # the -1 is very non-intuitive but correct
        self.distance_matrix = np.linalg.norm(self.coordinates - center, axis=1)

        # time.sleep(5) # wait for connection to open
        conn_timeout = 1000
        while not self.is_ready() and conn_timeout > 0:
            time.sleep(0.05)
            conn_timeout -= 1
        self.new_player()
        self.drawer = Game.Drawer(self.state_size, self.state_size)
        self.grid_drawer = Game.DrawerGrid(grid_size=11, width=1000, height=1000)

        self.last_state = None
        self.current_state = None
        self.frame_buffer = deque(maxlen=self.n_frames_state)

        conn_timeout = 1000
        self.action({"input_x": 0, "input_y": 0})
        while conn_timeout > 0:
            time.sleep(0.05)
            state = self.get_state_reward()

            if state is not None:
                break

            conn_timeout -= 1

        # get first samples from the simulation
        for _ in range(1, self.n_frames_state):
            self.action({"input_x": 0, "input_y": 0})
            self.get_state_reward()  # load the frame buffer and current state

        # second action to get the first sample to be really consumed
        self.action({"input_x": 0, "input_y": 0})
        self.get_state_reward()

        self.animated = False


    def get_identifier(self):
        return self.port

    def is_terminal(self):
        return self.current_state["is_terminal"]

    def is_ready(self):
        return self.client.ws.sock is not None and self.client.ws.sock.connected is True

    class MessageType:
        ERROR = -1
        IS_ALIVE = 0
        ALIVE = 1
        GUI_REFRESH = 2
        INPUT_CONTROLLER = 3
        NEW_PLAYER = 4
        NEW_FRAME = 5

    class MessageClientSenderStub:
        @staticmethod
        def new_player(player_name):
            return {"message_intention": 4, "player_name": player_name}

        @staticmethod
        def alive():
            return {"message_intention": Game.MessageType.ALIVE}

        @staticmethod
        def input_controller(input_x, input_y):
            return {"message_intention": Game.MessageType.INPUT_CONTROLLER, "input_x": input_x, "input_y": input_y}

    def new_player(self):
        message = Game.MessageClientSenderStub.new_player(self.name)
        message = json.dumps(message)
        self.client.send(message)

    class DrawerGrid:
        def __init__(self, grid_size, width, height):
            self.grid_size = grid_size
            self.pallet_grid = np.zeros((grid_size, grid_size))
            self.wall_grid = np.zeros((grid_size, grid_size))
            self._width = width
            self._height = height

        def dump_to_frame(self, frame_dump):
            main_coordinates = frame_dump['main_coordinates']
            scope = frame_dump['scope']
            rectangles = frame_dump['rectangles']

            self.pallet_grid = np.zeros((self.grid_size, self.grid_size))
            self.wall_grid = np.zeros((self.grid_size, self.grid_size))

            borders = frame_dump['borders']

            for border in borders:
                border["pos_x"] = border.pop("up_left_x") + border["width"] / 2
                border["pos_y"] = border.pop("up_left_y") + border["height"] / 2
                self.draw_rectangle(border, main_coordinates, scope, type="wall")

            for rectangle in rectangles:
                if 'name' not in rectangle:
                    self.draw_rectangle(rectangle, main_coordinates, scope, type="pallet")

            return {"walls": np.copy(self.wall_grid), "pallets": np.copy(self.pallet_grid)}

        def draw_rectangle(self, rectangle, main_coordinates, scope, type):
            global_pos_x = rectangle['pos_x']
            global_pos_y = rectangle['pos_y']

            if 'size' in rectangle:
                local_size = rectangle['size']
                width_on_screen = self.local_scale_to_screen_scale(local_size, scope)
                height_on_screen = self.local_scale_to_screen_scale(local_size, scope)
            elif 'width' in rectangle and 'height' in rectangle:
                local_width = rectangle['width']
                local_height = rectangle['height']
                width_on_screen = self.local_scale_to_screen_scale(local_width, scope)
                height_on_screen = self.local_scale_to_screen_scale(local_height, scope)

            (local_pos_x, local_pos_y) = self.coordinates_global_to_local((global_pos_x, global_pos_y),
                                                                          main_coordinates, scope)
            (screen_pos_x, screen_pos_y) = self.coordinates_local_to_screen((local_pos_x, local_pos_y), scope)

            index_x = int(screen_pos_x / (self.grid_size*self._width))
            index_y = int(screen_pos_y / (self.grid_size*self._height))

            def is_point_in(px, py, center_x, center_y, w, h):
                print((px, py, center_x, center_y, w, h))
                return center_x - w/2 <= px <= center_x + w/2 and center_y - h/2 <= py <= center_y + h/2

            if type == "wall":
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if is_point_in(px=(i+0.5)*self._width,
                                    py=(i+0.5)*self._height,
                                    center_x=screen_pos_x,
                                    center_y=screen_pos_y,
                                    w=width_on_screen,
                                    h=height_on_screen):
                            print((i, j))
                            self.wall_grid[index_x, index_y] += 1
            elif type == "pallet":
                self.pallet_grid[index_x, index_y] += 1


        def is_rect_on_screen(self, center_x, center_y, size):
            left_top = (center_x - size / 2, center_y - size / 2)
            right_down = (center_x + size / 2, center_y + size / 2)

            return self.is_point_on_screen(left_top[0], left_top[1]) or \
                   self.is_point_on_screen(right_down[0], left_top[1]) or \
                   self.is_point_on_screen(right_down[0], right_down[1]) or \
                   self.is_point_on_screen(left_top[0], right_down[1])

        def is_point_on_screen(self, pos_x, pos_y):
            return 0 <= pos_x <= self._width and 0 <= pos_y <= self._height

            # transform global map coordinates to local map coordinates with both reference frame on upward left

        """
            Input: 
                - global coordinates: coordinates in the main reference frame
                - main_coordinates: coordinates of the player's center
                - scope: field of view of the player

            NB. There is no scale transform yet
        """

        def coordinates_global_to_local(self, global_coordinates, main_coordinates, scope):
            return (global_coordinates[0] - main_coordinates[0] + scope,
                    global_coordinates[1] - main_coordinates[1] + scope)

        def coordinates_local_to_screen(self, local_coordinates, scope):
            return ((local_coordinates[0] / scope) * (self._width / 2),
                    (local_coordinates[1] / scope) * (self._height / 2))

        def local_scale_to_screen_scale(self, local_length, scope):
            # WIDTH AND HEIGHT MUST BE THE SAME
            return (local_length / scope) * (self._width / 2)

    class Drawer:
        def __init__(self, width, height):
            self._width = width
            self._height = height
            self.array = np.zeros((height, width))

        def flush(self):
            self.array.fill(0)

        def fill_rectangle(self, x, y, w, h, color):
            cv2.rectangle(self.array, pt1=(int(x), int(y)), pt2=(int(x + w), int(y + h)), color=color, thickness=-1)

        # from dump generate new frame
        def dump_to_frame(self, frame_dump):
            main_coordinates = frame_dump['main_coordinates']
            scope = frame_dump['scope']
            rectangles = frame_dump['rectangles']

            self.flush()  # fill with black

            borders = frame_dump['borders']

            for border in borders:
                border["color"] = (0.25,)
                border["pos_x"] = border.pop("up_left_x") + border["width"] / 2
                border["pos_y"] = border.pop("up_left_y") + border["height"] / 2
                self.draw_rectangle(border, main_coordinates, scope)

            priority_rectangles = []  # rectangles to draw at last
            for rectangle in rectangles:
                if 'name' in rectangle:
                    priority_rectangles.append(rectangle)
                    continue

                rectangle["color"] = (1,)
                self.draw_rectangle(rectangle, main_coordinates, scope)

            for rectangle in priority_rectangles:
                rectangle["color"] = (0.5,)
                self.draw_rectangle(rectangle, main_coordinates, scope)

            return np.copy(self.array)#np.copy(cv2.merge((self.array, self.array, self.array)))

        def draw_rectangle(self, rectangle, main_coordinates, scope):
            color = rectangle['color']
            global_pos_x = rectangle['pos_x']
            global_pos_y = rectangle['pos_y']

            if 'size' in rectangle:
                local_size = rectangle['size']
                width_on_screen = self.local_scale_to_screen_scale(local_size, scope)
                height_on_screen = self.local_scale_to_screen_scale(local_size, scope)
            elif 'width' in rectangle and 'height' in rectangle:
                local_width = rectangle['width']
                local_height = rectangle['height']
                width_on_screen = self.local_scale_to_screen_scale(local_width, scope)
                height_on_screen = self.local_scale_to_screen_scale(local_height, scope)

            (local_pos_x, local_pos_y) = self.coordinates_global_to_local((global_pos_x, global_pos_y),
                                                                          main_coordinates, scope)
            (screen_pos_x, screen_pos_y) = self.coordinates_local_to_screen((local_pos_x, local_pos_y), scope)

            # draw the rectangle
            self.fill_rectangle(x=screen_pos_x - width_on_screen / 2,
                                y=screen_pos_y - height_on_screen / 2,
                                w=width_on_screen,
                                h=height_on_screen,
                                color=color)

        def is_rect_on_screen(self, center_x, center_y, size):
            left_top = (center_x - size / 2, center_y - size / 2)
            right_down = (center_x + size / 2, center_y + size / 2)

            return self.is_point_on_screen(left_top[0], left_top[1]) or \
                   self.is_point_on_screen(right_down[0], left_top[1]) or \
                   self.is_point_on_screen(right_down[0], right_down[1]) or \
                   self.is_point_on_screen(left_top[0], right_down[1])

        def is_point_on_screen(self, pos_x, pos_y):
            return 0 <= pos_x <= self._width and 0 <= pos_y <= self._height

        # transform global map coordinates to local map coordinates with both reference frame on upward left
        """
            Input: 
                - global coordinates: coordinates in the main reference frame
                - main_coordinates: coordinates of the player's center
                - scope: field of view of the player

            NB. There is no scale transform yet
        """

        def coordinates_global_to_local(self, global_coordinates, main_coordinates, scope):
            return (global_coordinates[0] - main_coordinates[0] + scope,
                    global_coordinates[1] - main_coordinates[1] + scope)

        def coordinates_local_to_screen(self, local_coordinates, scope):
            return ((local_coordinates[0] / scope) * (self._width / 2),
                    (local_coordinates[1] / scope) * (self._height / 2))

        def local_scale_to_screen_scale(self, local_length, scope):
            # WIDTH AND HEIGHT MUST BE THE SAME
            return (local_length / scope) * (self._width / 2)

    def act_and_observe(self, action, max_steps=None):
        self.action(action)
        current_state = self.get_state_reward(max_steps)
        ret = {"identifier": self.port,
               "state": self.last_state["state"],
               "action": action,
               "new_state": current_state["state"],
               "last_frame": current_state["last_frame"],
               "reward": current_state["reward"],
               "is_terminal": current_state["is_terminal"],
               "remaining_pallets": current_state["remaining_pallets"]}
        return ret

    def get_current_state(self):
        return self.current_state

    def get_state_reward(self, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps

        dump = self.client.receive(block=True)
        dump = json.loads(dump)
        if "main_coordinates" not in dump["frame_dump"]:
            return None

        frame = self.drawer.dump_to_frame(dump["frame_dump"])

        if len(dump["frame_dump"]["rectangles"]) == 0:
            return None

        size = None

        """frame_flat = frame.flatten()
        mask = frame_flat == 1
        if mask.sum() == 0:
            reward = 0
        else:
            target = mask * self.distance_matrix
            target[target == 0] = np.infty
            closest = np.argmin(target)
            coordinate_closest = self.coordinates[closest]
            target = coordinate_closest

            (vel_x, vel_y) = dump["frame_dump"]["velocity"]

            def coordinate_to_input(x, y):
                distance_from_center = ((x - self.state_size / 2) ** 2 + (y - self.state_size / 2) ** 2) ** 0.5

                max_distance = self.state_size / 4  # assume square screen
                if distance_from_center >= max_distance:
                    distance_from_center = max_distance

                if (x - self.state_size / 2) == 0:  # special case
                    angle = np.pi / 2
                else:
                    angle = np.arctan((y - self.state_size / 2) / (x - self.state_size / 2))

                projection_x = np.cos(angle) * distance_from_center
                projection_y = np.sin(angle) * distance_from_center

                if (x - self.state_size / 2) < 0:  # this is w.r.t x, not as usual, the reference frame is weird
                    projection_x = - projection_x
                    projection_y = - projection_y

                input_x = projection_x / max_distance
                input_y = projection_y / max_distance

                return (input_x, input_y)

            (input_x, input_y) = coordinate_to_input(target[1], target[0])


            reward = np.dot(np.array([input_x, input_y]), np.array([vel_x, vel_y]))"""

        """for rect in dump["frame_dump"]["rectangles"]:
            if "name" in rect and rect["name"] == self.name:
                if self.last_state is None or "size" not in self.last_state:  # first state
                    reward = 0
                    self.prev_pellets = dump["frame_dump"]["n_pallets"]

                    # reward = (rect["size"] - self.last_state["size"])
                    # reward = (rect["size"] - self.last_state["size"])* 50
                    # print((rect["size"] - self.last_state["size"]))
                    # print(reward)
                    # reward = ((rect["size"] - self.last_state["size"]) + 1)**2

                    size = rect["size"]

                    # reward = rect["size"]
                    break"""

        if self.prev_pellets is not None:
            if self.prev_pellets != dump["frame_dump"]["n_pallets"]:
                reward = self.prev_pellets - dump["frame_dump"]["n_pallets"]
            else:
                reward = -1
        else:
            reward = 0

        self.prev_pellets = dump["frame_dump"]["n_pallets"]

        def is_player_out(player_position, world_size):
            player_x = player_position[0]
            player_y = player_position[1]

            world_width = world_size[0]
            world_height = world_size[1]

            return not (0 <= player_x <= world_width) or not (0 <= player_y <= world_height)

        is_terminal = False

        """if size <= 0.1:
            is_terminal = True
            reward -= 100"""

        if is_player_out(player_position=dump["frame_dump"]["main_coordinates"],
                         world_size=dump["frame_dump"]["world_size"]):
            pass
            # is_terminal = True
            # reward -= 100000  # reward penalty for getting out

        if dump["frame_dump"]["n_pallets"] == 0:
            is_terminal = True
            #reward += 100000

        if self.life > max_steps and dump["frame_dump"]["n_pallets"] != 0:
            is_terminal = True

        self.life += 1

        if is_terminal is True:
            is_terminal = np.array(1.0)
        else:
            is_terminal = np.array(0.0)

        self.frame_buffer.append(frame)
        self.last_state = self.current_state
        self.current_state = {"identifier": self.port, "state": list(self.frame_buffer), "last_frame": frame,
                              "size": size, "reward": reward, "is_terminal": is_terminal,
                              "remaining_pallets": dump["frame_dump"]["n_pallets"]}
        return self.current_state

    def animate(self, frame=None, x_window=0, y_window=0):
        import time
        if frame is None:
            dump = self.client.receive(block=True)
            dump = json.loads(dump)['frame_dump']

            if len(dump['rectangles']) > 0:
                return

            frame = self.drawer.dump_to_frame(dump)

        t = time.time()
        # print("{:.3f}".format((time.time() - t)*1000))
        if self.animated is False:
            cv2.namedWindow(str(self.port))

        if isinstance(frame, list):
            cv2.imshow(str(self.port), frame[len(frame)-1])
        else:
            cv2.imshow(str(self.port), frame)

        if x_window is not None and y_window is not None:
            cv2.moveWindow(str(self.port), x_window, y_window)
        cv2.waitKey(1)

        self.animated = True

    def action(self, action):
        input_x = action["input_x"]
        input_y = action["input_y"]
        message = Game.MessageClientSenderStub.input_controller(input_x, input_y)
        message = json.dumps(message)
        self.client.send(message)


class Client:

    def __init__(self, host, port):
        self.ws = websocket.WebSocketApp("ws://{}:{}".format(host, port),
                                         on_message=lambda ws, message: self.on_message(ws, message),
                                         on_error=lambda ws, error: self.on_error(ws, error),
                                         on_close=lambda ws: self.on_close(ws))

        self.ws.on_open = lambda ws: self.on_open(ws)

        self.message_queue = queue.Queue()

    def start(self):
        # https://stackoverflow.com/questions/29145442/threaded-non-blocking-websocket-client
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def send(self, message):
        # print("Message sent: {}".format(message))
        self.ws.send(message)

    def receive(self, block=True, timeout=None):
        try:
            return self.message_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def on_message(self, ws, message):
        # print("Message received: {}".format(message))
        self.message_queue.put_nowait(message)

    def on_error(self, ws, error):
        print("SOCKET ERROR: {}".format(error))

    def on_close(self, ws):
        pass
        # print("Client closed")

    def on_open(self, ws):
        pass
        # print("Connection opened")


if __name__ == "__main__":
    # size of state (frame)
    state_size = 200
    game_manager = GamesManager(state_size=state_size, n_pallets=1000)

    # number of games in //
    n_games = 10
    # request the games to the game engine
    game_manager.request_new_games(n_games=n_games, name="test")
    # baseline agent
    import BaselineClosest

    actor = BaselineClosest.BaselineClosest(state_size=state_size)
    # pretty showing
    n_columns_windows = 7
    # main agents loop
    while True:

        states_rewards = game_manager.get_current_states()
        for i, state_reward in enumerate(states_rewards):
            if state_reward["identifier"] in game_manager.games:
                game = game_manager.games[state_reward["identifier"]]
            else:
                continue
            x_window = i % n_columns_windows
            y_window = int(i / n_columns_windows)
            x_window = (x_window * (state_size + 50)) + 50
            y_window = (y_window * (state_size + 50)) + 50
            game.animate(state_reward["state"], x_window=x_window, y_window=y_window)

        buffer_actions = {}
        for state_reward in states_rewards:
            if state_reward["identifier"] not in game_manager.games:
                continue
            (x, y) = actor.action(state_reward)
            print(state_reward["identifier"])
            buffer_actions[state_reward["identifier"]] = {"input_x": x, "input_y": y}
        # game_manager.actions({port: {"input_x": x, "input_y": y} for port in list(game_manager.games.keys())})
        # game_manager.actions(buffer_actions)
        states_rewards = game_manager.act_and_observe(buffer_actions)

        """for state_reward in states_rewards:
            if state_reward["is_terminal"] is True:
                cv2.destroyWindow(state_reward["identifier"])"""
        # r = game_manager.get_states_rewards()
        time.sleep(0.015)
        # print(r)



