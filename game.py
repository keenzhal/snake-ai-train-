import pygame
import random
import numpy as np

BLOCK_SIZE = 20
SPEED = 40

class SnakeGame:
    def __init__(self, w=400, h=400, panel_width=260, left_logs_width=340, speed=SPEED):
        pygame.init()
        self.w = w
        self.h = h
        self.panel_width = panel_width
        self.left_logs_width = left_logs_width
        self.game_origin_x = self.left_logs_width
        self.speed = speed
        total_width = self.left_logs_width + self.w + self.panel_width
        self.display = pygame.display.set_mode((total_width, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.training_info = {}
        self.running = True
        self.reset()

    def reset(self):
        self.direction = "RIGHT"
        self.head = [self.w//2, self.h//2]
        self.snake = [self.head[:],
                      [self.head[0]-BLOCK_SIZE, self.head[1]],
                      [self.head[0]-2*BLOCK_SIZE, self.head[1]]]
        self.max_cells = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        self.score = 0
        self.steps_since_food = 0
        self.food = None
        self._place_food()
        return self.get_state()

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        return self.running

    def close(self):
        pygame.quit()

    def set_training_info(self, info):
        self.training_info = info or {}

    def _place_food(self):
        occupied = {tuple(p) for p in self.snake}
        free_cells = []
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                if (x, y) not in occupied:
                    free_cells.append((x, y))

        if not free_cells:
            self.food = None
            return

        x, y = random.choice(free_cells)
        self.food = [x, y]

    def step(self, action):
        if not self.running:
            return self.get_state(), 0, True

        reward = 0
        done = False
        if action not in (0, 1, 2):
            action = 0

        self.steps_since_food += 1
        self._move(action)
        self.snake.insert(0, self.head[:])

        if self._collision():
            reward = -10
            done = True
        else:
            if self.food is not None and self.head == self.food:
                self.score += 1
                reward = 10
                self.steps_since_food = 0
                self._place_food()
            else:
                self.snake.pop()

            if len(self.snake) >= self.max_cells:
                reward += 50
                done = True
            elif not self._get_valid_actions():
                reward -= 10
                done = True

        self._update_ui()
        self.clock.tick(self.speed)

        return self.get_state(), reward, done

    def _collision(self):
        if (self.head[0] >= self.w or self.head[0] < 0 or
            self.head[1] >= self.h or self.head[1] < 0):
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        self.direction = self._direction_after_action(action, self.direction)
        x, y = self.head
        if self.direction == "RIGHT":
            x += BLOCK_SIZE
        elif self.direction == "LEFT":
            x -= BLOCK_SIZE
        elif self.direction == "DOWN":
            y += BLOCK_SIZE
        elif self.direction == "UP":
            y -= BLOCK_SIZE

        self.head = [x, y]

    def _direction_after_action(self, action, current_direction):
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = directions.index(current_direction)
        if action == 1:  # right turn
            return directions[(idx + 1) % 4]
        if action == 2:  # left turn
            return directions[(idx - 1) % 4]
        return directions[idx]

    def _next_head_for_action(self, action):
        direction = self._direction_after_action(action, self.direction)
        x, y = self.head
        if direction == "RIGHT":
            x += BLOCK_SIZE
        elif direction == "LEFT":
            x -= BLOCK_SIZE
        elif direction == "DOWN":
            y += BLOCK_SIZE
        elif direction == "UP":
            y -= BLOCK_SIZE
        return [x, y]

    def _action_would_collide(self, action):
        next_head = self._next_head_for_action(action)
        if (
            next_head[0] >= self.w
            or next_head[0] < 0
            or next_head[1] >= self.h
            or next_head[1] < 0
        ):
            return True

        would_grow = self.food is not None and next_head == self.food
        future_snake = [next_head[:]] + [p[:] for p in self.snake]
        if not would_grow:
            future_snake.pop()

        return next_head in future_snake[1:]

    def _get_valid_actions(self):
        valid = []
        for action in (0, 1, 2):
            if not self._action_would_collide(action):
                valid.append(action)
        return valid

    def get_state(self):
        head = self.head
        food = self.food if self.food is not None else self.head
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"

        state = [
            (dir_r and self._collision_point(point_r)) or
            (dir_l and self._collision_point(point_l)) or
            (dir_u and self._collision_point(point_u)) or
            (dir_d and self._collision_point(point_d)),

            (dir_u and self._collision_point(point_r)) or
            (dir_d and self._collision_point(point_l)) or
            (dir_l and self._collision_point(point_u)) or
            (dir_r and self._collision_point(point_d)),

            (dir_d and self._collision_point(point_r)) or
            (dir_u and self._collision_point(point_l)) or
            (dir_r and self._collision_point(point_u)) or
            (dir_l and self._collision_point(point_d)),

            dir_l, dir_r, dir_u, dir_d,

            food[0] < head[0],
            food[0] > head[0],
            food[1] < head[1],
            food[1] > head[1]
        ]

        return np.array(state, dtype=int)

    def _collision_point(self, point):
        if (point[0] >= self.w or point[0] < 0 or
            point[1] >= self.h or point[1] < 0):
            return True
        if point in self.snake:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0, 0, 0))

        game_rect = pygame.Rect(self.game_origin_x, 0, self.w, self.h)
        pygame.draw.rect(self.display, (0, 0, 0), game_rect)
        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                (0, 255, 0),
                pygame.Rect(pt[0] + self.game_origin_x, pt[1], BLOCK_SIZE, BLOCK_SIZE),
            )
        if self.food is not None:
            pygame.draw.rect(
                self.display,
                (255, 0, 0),
                pygame.Rect(self.food[0] + self.game_origin_x, self.food[1], BLOCK_SIZE, BLOCK_SIZE),
            )
        score_text = self.small_font.render(f"Score: {self.score}", True, (220, 220, 220))
        self.display.blit(score_text, (self.game_origin_x + 8, 8))

        if self.left_logs_width > 0:
            pygame.draw.line(
                self.display,
                (55, 55, 55),
                (self.game_origin_x, 0),
                (self.game_origin_x, self.h),
                2,
            )
            self._draw_left_logs_panel()
        pygame.draw.line(
            self.display,
            (55, 55, 55),
            (self.game_origin_x + self.w, 0),
            (self.game_origin_x + self.w, self.h),
            2,
        )
        self._draw_training_panel()

        pygame.display.flip()

    def _draw_training_panel(self):
        if self.panel_width <= 0:
            return

        panel_x = self.game_origin_x + self.w + 2
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width - 2, self.h)
        pygame.draw.rect(self.display, (18, 18, 18), panel_rect)

        x0 = panel_x + 10
        y0 = 12

        lines = [
            ("Training", ""),
            ("Game", self.training_info.get("game", 1)),
            ("Score", self.training_info.get("score", self.score)),
            ("Best", self.training_info.get("best_score", 0)),
            ("Average", self.training_info.get("avg_score", 0.0)),
            ("Step", self.training_info.get("step", 0)),
            ("Ep Reward", self.training_info.get("episode_reward", 0)),
            ("Target@Step", self.training_info.get("target_score", 0)),
            ("Delta", self.training_info.get("delta_to_target", 0)),
            ("Epsilon", self.training_info.get("epsilon", 0)),
            ("Memory", self.training_info.get("memory_size", 0)),
        ]

        for idx, (label, value) in enumerate(lines):
            if idx == 0:
                text = self.font.render(label, True, (235, 235, 235))
            else:
                text = self.small_font.render(f"{label}: {value}", True, (200, 200, 200))
            self.display.blit(text, (x0, y0 + idx * 24))

        history = self.training_info.get("score_history", [])
        chart_rect = pygame.Rect(x0, self.h - 170, self.panel_width - 24, 145)
        pygame.draw.rect(self.display, (10, 10, 10), chart_rect)
        pygame.draw.rect(self.display, (90, 90, 90), chart_rect, 1)

        chart_title = self.small_font.render("Score History (Live)", True, (190, 190, 190))
        self.display.blit(chart_title, (x0, self.h - 190))

        if len(history) >= 2:
            self._draw_score_chart(history, chart_rect)

    def _draw_left_logs_panel(self):
        logs = self.training_info.get("ai_logs", [])
        if not logs:
            return

        max_lines = 24
        x0 = 10
        y_start = 12
        panel_width = self.left_logs_width - 20
        line_height = 15
        max_by_height = max(1, (self.h - y_start - 12) // line_height)
        visible_logs = logs[: min(max_lines, max_by_height)]
        panel_height = 8 + len(visible_logs) * line_height

        rect = pygame.Rect(x0, y_start, panel_width, panel_height)
        pygame.draw.rect(self.display, (18, 18, 18), rect)
        pygame.draw.rect(self.display, (90, 90, 90), rect, 1)

        for idx, line in enumerate(visible_logs):
            color = self._log_color(str(line), idx)
            text = self.small_font.render(str(line), True, color)
            self.display.blit(text, (x0 + 5, y_start + 4 + idx * line_height))

    def _log_color(self, line, idx):
        if idx == 0 or line.startswith("AI Live Log"):
            return (170, 235, 255)
        if line.startswith("Danger"):
            return (255, 135, 135)
        if line.startswith("Dir "):
            return (145, 190, 255)
        if line.startswith("Food "):
            return (255, 205, 120)
        if line.startswith("Head:"):
            return (205, 205, 205)
        if line.startswith("Q(s)"):
            return (220, 190, 255)
        if line.startswith("Policy%"):
            return (130, 255, 180)
        if line.startswith("Next%"):
            return (90, 255, 160)
        if line.startswith("softmax") or line.startswith("eps-greedy") or line.startswith("e=clip"):
            return (145, 215, 255)
        if line.startswith("S:"):
            return (255, 235, 120)
        if line.startswith("R:"):
            return (120, 245, 255)
        if line.startswith("L:"):
            return (255, 165, 205)
        if line.startswith("eps="):
            return (175, 175, 175)
        if line.startswith("Explore chance"):
            return (255, 215, 140)
        if line.startswith("Action:"):
            return (255, 245, 170)
        return (190, 190, 190)

    def _draw_score_chart(self, history, rect):
        recent = history[-min(60, len(history)):]
        min_val = min(recent)
        max_val = max(recent)
        if max_val == min_val:
            max_val += 1

        points = []
        total = len(recent)
        for idx, val in enumerate(recent):
            x = rect.left + int(idx * (rect.width - 1) / max(1, total - 1))
            norm = (val - min_val) / (max_val - min_val)
            y = rect.bottom - 1 - int(norm * (rect.height - 1))
            points.append((x, y))

        pygame.draw.lines(self.display, (85, 220, 255), False, points, 2)

    def _direction_to_action(self, desired_direction):
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        current_idx = directions.index(self.direction)
        desired_idx = directions.index(desired_direction)
        delta = (desired_idx - current_idx) % 4

        if delta == 0:
            return 0
        if delta == 1:
            return 1
        if delta == 3:
            return 2

        # 180-degree turn is invalid in one step, ignore input.
        return None

    def run_manual(self):
        self.running = True
        key_to_direction = {
            pygame.K_w: "UP",
            pygame.K_a: "LEFT",
            pygame.K_s: "DOWN",
            pygame.K_d: "RIGHT",
            pygame.K_UP: "UP",
            pygame.K_LEFT: "LEFT",
            pygame.K_DOWN: "DOWN",
            pygame.K_RIGHT: "RIGHT",
        }

        while self.running:
            action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key in key_to_direction:
                        desired_direction = key_to_direction[event.key]
                        mapped_action = self._direction_to_action(desired_direction)
                        if mapped_action is not None:
                            action = mapped_action

            if not self.running:
                break

            _, _, done = self.step(action)
            if done:
                print(f"Game over. Score: {self.score}.")
                self.running = False

        self.close()


if __name__ == "__main__":
    game = SnakeGame(panel_width=0, left_logs_width=0, speed=12)
    game.run_manual()
