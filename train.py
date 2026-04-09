from collections import deque
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from agent import Agent, MAX_MEMORY
from game import SnakeGame

DATASET_DIR = Path("datasets")
TRANSITIONS_FILE = DATASET_DIR / "transitions.csv"
EPISODES_FILE = DATASET_DIR / "episodes.csv"
CHALLENGE_FILE = DATASET_DIR / "challenge_metrics.csv"
PLOTS_DIR = Path("plots")


def open_csv_writer(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    file = path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(header)
        file.flush()
    return file, writer


def state_to_list(state):
    return [int(v) for v in state.tolist()]


def build_ai_logs(state, decision, head, food):
    q = decision.get("q_values", [0.0, 0.0, 0.0])
    policy_probs = decision.get("policy_probs", [0.0, 0.0, 0.0])
    effective_probs = decision.get("effective_probs", [0.0, 0.0, 0.0])
    action_names = ["STRAIGHT", "RIGHT", "LEFT"]
    action_idx = int(decision.get("action", 0))
    mode = decision.get("mode", "unknown")
    epsilon = float(decision.get("epsilon", 0.0))
    explore_rate = float(decision.get("explore_rate", 0.0))
    roll = int(decision.get("roll", 0))
    state_list = [int(v) for v in state.tolist()]
    clipped_next_probs = [max(0.0, min(1.0, float(p))) for p in effective_probs]

    def prob_bar(prob, width=16):
        filled = int(round(prob * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    logs = [
        "AI Live Log",
        f"State: {state_list}",
        f"Danger S/R/L: {state_list[0]}/{state_list[1]}/{state_list[2]}",
        f"Dir L/R/U/D: {state_list[3]}/{state_list[4]}/{state_list[5]}/{state_list[6]}",
        f"Food L/R/U/D: {state_list[7]}/{state_list[8]}/{state_list[9]}/{state_list[10]}",
        f"Head: {head}  Food: {food}",
        f"Q(s)=[{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}]",
        (
            "Policy% S/R/L: "
            f"{policy_probs[0] * 100:.1f}/{policy_probs[1] * 100:.1f}/{policy_probs[2] * 100:.1f}"
        ),
        (
            "Next% S/R/L: "
            f"{effective_probs[0] * 100:.1f}/{effective_probs[1] * 100:.1f}/{effective_probs[2] * 100:.1f}"
        ),
        "softmax: p_i=exp(q_i-m)/sum_j exp(q_j-m), m=max(q)",
        "eps-greedy: P_i=e/3 + (1-e)*I(i=argmax q)",
        f"e=clip(eps/200, 0, 1)={explore_rate:.3f}",
        f"S: {clipped_next_probs[0] * 100:5.1f}% {prob_bar(clipped_next_probs[0])}",
        f"R: {clipped_next_probs[1] * 100:5.1f}% {prob_bar(clipped_next_probs[1])}",
        f"L: {clipped_next_probs[2] * 100:5.1f}% {prob_bar(clipped_next_probs[2])}",
        f"eps={epsilon:.1f}, roll={roll}, mode={mode}",
        f"Explore chance: {explore_rate * 100:.1f}%",
        f"Action: {action_names[action_idx]} ({action_idx})",
    ]
    return logs


def load_episode_history(path):
    score_history = []
    avg_history = []
    max_game_id = 0

    if not path.exists():
        return {
            "score_history": score_history,
            "avg_history": avg_history,
            "n_games": 0,
            "best_score": 0,
            "total_score": 0,
        }

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                game_id = int(row.get("game", 0))
                score = int(float(row.get("score", 0)))
                avg_score = float(row.get("avg_score", 0.0))
            except (TypeError, ValueError):
                continue

            if game_id > max_game_id:
                max_game_id = game_id
            score_history.append(score)
            avg_history.append(avg_score)

    total_score = int(sum(score_history))
    best_score = max(score_history) if score_history else 0
    n_games = max(max_game_id, len(score_history))

    return {
        "score_history": score_history,
        "avg_history": avg_history,
        "n_games": n_games,
        "best_score": best_score,
        "total_score": total_score,
    }


def load_transitions_into_memory(agent, path, state_size):
    if not path.exists():
        return 0, {}

    state_cols = [f"state_{i}" for i in range(state_size)]
    next_state_cols = [f"next_state_{i}" for i in range(state_size)]

    loaded = 0
    restored_memory = deque(maxlen=MAX_MEMORY)
    best_score_exact_step = {}

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                state = np.array([int(row[col]) for col in state_cols], dtype=int)
                next_state = np.array([int(row[col]) for col in next_state_cols], dtype=int)
                action = int(row["action"])
                reward = float(row["reward"])
                done = bool(int(row["done"]))
                step = int(row["step"])
                score = int(float(row["score"]))
            except (KeyError, TypeError, ValueError):
                continue

            restored_memory.append((state, action, reward, next_state, done))
            if score > best_score_exact_step.get(step, -1):
                best_score_exact_step[step] = score
            loaded += 1

    agent.memory = restored_memory
    return loaded, best_score_exact_step


def build_target_curve(best_score_exact_step):
    if not best_score_exact_step:
        return {}

    max_step = max(best_score_exact_step.keys())
    curve = {}
    best_so_far = 0
    for step in range(1, max_step + 1):
        best_so_far = max(best_so_far, best_score_exact_step.get(step, 0))
        curve[step] = best_so_far
    return curve


def get_target_score_for_step(step, target_curve):
    if not target_curve:
        return 0

    max_step = max(target_curve.keys())
    if step <= max_step:
        return int(target_curve.get(step, 0))
    return int(target_curve[max_step])


def load_challenge_history(path):
    history = []
    if not path.exists():
        return history

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                history.append(
                    {
                        "game": int(row["game"]),
                        "steps": int(row["steps"]),
                        "score": int(row["score"]),
                        "target_score": int(row["target_score"]),
                        "delta": int(row["delta"]),
                        "efficiency": float(row["efficiency"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

    return history


def save_training_plot(score_history, avg_history):
    if not score_history:
        return None

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PLOTS_DIR / f"training_scores_{timestamp}.png"

    width, height = 1100, 700
    margin_left, margin_right = 90, 50
    margin_top, margin_bottom = 80, 100
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_bottom = plot_top + plot_height

    surface = pygame.Surface((width, height))
    surface.fill((245, 245, 245))

    font_title = pygame.font.SysFont("consolas", 30)
    font_text = pygame.font.SysFont("consolas", 20)
    font_small = pygame.font.SysFont("consolas", 16)

    all_values = list(score_history) + list(avg_history)
    min_val = min(all_values)
    max_val = max(all_values)
    if max_val == min_val:
        max_val += 1

    pygame.draw.line(surface, (60, 60, 60), (plot_left, plot_top), (plot_left, plot_bottom), 2)
    pygame.draw.line(surface, (60, 60, 60), (plot_left, plot_bottom), (plot_left + plot_width, plot_bottom), 2)

    y_ticks = 5
    for i in range(y_ticks + 1):
        y = plot_top + int(i * plot_height / y_ticks)
        pygame.draw.line(surface, (215, 215, 215), (plot_left, y), (plot_left + plot_width, y), 1)
        value = max_val - (i / y_ticks) * (max_val - min_val)
        label = font_small.render(f"{value:.1f}", True, (70, 70, 70))
        surface.blit(label, (plot_left - 65, y - 8))

    n = len(score_history)
    x_ticks = min(10, n)
    if x_ticks > 1:
        for i in range(x_ticks):
            idx = int(i * (n - 1) / (x_ticks - 1))
            x = plot_left + int(idx * plot_width / max(1, n - 1))
            pygame.draw.line(surface, (215, 215, 215), (x, plot_top), (x, plot_bottom), 1)
            label = font_small.render(str(idx + 1), True, (70, 70, 70))
            surface.blit(label, (x - 8, plot_bottom + 8))

    def make_points(values):
        points = []
        total = len(values)
        for idx, val in enumerate(values):
            x = plot_left + int(idx * plot_width / max(1, total - 1))
            norm = (val - min_val) / (max_val - min_val)
            y = plot_bottom - int(norm * plot_height)
            points.append((x, y))
        return points

    score_points = make_points(score_history)
    avg_points = make_points(avg_history)

    if len(score_points) >= 2:
        pygame.draw.lines(surface, (40, 130, 230), False, score_points, 3)
    if len(avg_points) >= 2:
        pygame.draw.lines(surface, (230, 120, 40), False, avg_points, 3)

    title = font_title.render("Snake AI Training Progress", True, (20, 20, 20))
    surface.blit(title, (plot_left, 20))

    xlabel = font_text.render("Episode", True, (40, 40, 40))
    ylabel = font_text.render("Score", True, (40, 40, 40))
    surface.blit(xlabel, (plot_left + plot_width // 2 - 40, height - 50))
    surface.blit(ylabel, (20, plot_top + plot_height // 2 - 10))

    legend_y = 55
    pygame.draw.line(surface, (40, 130, 230), (plot_left + 480, legend_y), (plot_left + 530, legend_y), 4)
    surface.blit(font_small.render("Score", True, (40, 40, 40)), (plot_left + 540, legend_y - 10))
    pygame.draw.line(surface, (230, 120, 40), (plot_left + 620, legend_y), (plot_left + 670, legend_y), 4)
    surface.blit(font_small.render("Average Score", True, (40, 40, 40)), (plot_left + 680, legend_y - 10))

    footer = font_small.render(
        f"Episodes: {len(score_history)}  |  Best: {max(score_history)}  |  Last Avg: {avg_history[-1]:.2f}",
        True,
        (40, 40, 40),
    )
    surface.blit(footer, (plot_left, height - 25))

    pygame.image.save(surface, str(output_path))
    return output_path


def save_step_target_plot(target_curve, challenge_history):
    if not target_curve and not challenge_history:
        return None

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PLOTS_DIR / f"step_target_{timestamp}.png"

    width, height = 1100, 700
    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 80, 90
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_bottom = plot_top + plot_height

    surface = pygame.Surface((width, height))
    surface.fill((246, 246, 246))

    font_title = pygame.font.SysFont("consolas", 28)
    font_text = pygame.font.SysFont("consolas", 18)
    font_small = pygame.font.SysFont("consolas", 15)

    points = challenge_history[-300:]
    max_step_points = max([p["steps"] for p in points], default=0)
    max_step_curve = max(target_curve.keys(), default=0)
    max_step = max(max_step_points, max_step_curve, 1)

    max_score_points = max([p["score"] for p in points], default=0)
    max_score_curve = max(target_curve.values(), default=0)
    max_score = max(max_score_points, max_score_curve, 1)

    pygame.draw.line(surface, (60, 60, 60), (plot_left, plot_top), (plot_left, plot_bottom), 2)
    pygame.draw.line(surface, (60, 60, 60), (plot_left, plot_bottom), (plot_left + plot_width, plot_bottom), 2)

    for i in range(6):
        y = plot_top + int(i * plot_height / 5)
        pygame.draw.line(surface, (215, 215, 215), (plot_left, y), (plot_left + plot_width, y), 1)
        value = max_score - (i / 5) * max_score
        lbl = font_small.render(f"{value:.1f}", True, (70, 70, 70))
        surface.blit(lbl, (plot_left - 65, y - 8))

    for i in range(6):
        x = plot_left + int(i * plot_width / 5)
        pygame.draw.line(surface, (215, 215, 215), (x, plot_top), (x, plot_bottom), 1)
        value = int((i / 5) * max_step)
        lbl = font_small.render(str(value), True, (70, 70, 70))
        surface.blit(lbl, (x - 10, plot_bottom + 8))

    if target_curve:
        curve_points = []
        for step in range(1, max(target_curve.keys()) + 1):
            x = plot_left + int(step * plot_width / max_step)
            y = plot_bottom - int(target_curve.get(step, 0) * plot_height / max_score)
            curve_points.append((x, y))
        if len(curve_points) >= 2:
            pygame.draw.lines(surface, (230, 120, 40), False, curve_points, 3)

    for p in points:
        x = plot_left + int(p["steps"] * plot_width / max_step)
        y = plot_bottom - int(p["score"] * plot_height / max_score)
        color = (35, 150, 240) if p["delta"] >= 0 else (200, 90, 90)
        pygame.draw.circle(surface, color, (x, y), 3)

    title = font_title.render("Self-Competition: Score At Step Budget", True, (20, 20, 20))
    surface.blit(title, (plot_left, 22))
    surface.blit(font_text.render("X: Steps in episode", True, (40, 40, 40)), (plot_left, height - 48))
    surface.blit(font_text.render("Y: Score", True, (40, 40, 40)), (25, plot_top + plot_height // 2 - 10))

    pygame.draw.line(surface, (230, 120, 40), (plot_left + 620, 54), (plot_left + 665, 54), 4)
    surface.blit(font_small.render("Best historical target curve", True, (40, 40, 40)), (plot_left + 675, 44))
    pygame.draw.circle(surface, (35, 150, 240), (plot_left + 626, 80), 4)
    surface.blit(font_small.render("Episode result (blue met target)", True, (40, 40, 40)), (plot_left + 675, 72))

    pygame.image.save(surface, str(output_path))
    return output_path


def train():
    game = SnakeGame()
    agent = Agent()
    load_info = agent.load()
    loaded_games_before_csv_sync = agent.n_games
    state_size = len(game.get_state())

    transition_header = [
        "timestamp",
        "game",
        "step",
        "score",
        "action",
        "reward",
        "done",
    ] + [f"state_{i}" for i in range(state_size)] + [f"next_state_{i}" for i in range(state_size)]

    episode_header = [
        "timestamp",
        "game",
        "score",
        "best_score",
        "avg_score",
        "steps",
        "episode_reward",
        "epsilon",
        "memory_size",
    ]

    challenge_header = [
        "timestamp",
        "game",
        "steps",
        "score",
        "target_score",
        "delta",
        "efficiency",
    ]

    episode_history = load_episode_history(EPISODES_FILE)
    score_history = episode_history["score_history"]
    avg_history = episode_history["avg_history"]
    best_score = episode_history["best_score"]
    total_score = episode_history["total_score"]

    if episode_history["n_games"] > agent.n_games:
        agent.n_games = episode_history["n_games"]

    checkpoint_extra = load_info.get("extra_state", {}) if isinstance(load_info, dict) else {}
    if not score_history and checkpoint_extra:
        total_score = int(checkpoint_extra.get("total_score", total_score))
        best_score = int(checkpoint_extra.get("best_score", best_score))

    loaded_transitions, best_score_exact_step = load_transitions_into_memory(agent, TRANSITIONS_FILE, state_size)
    target_curve = build_target_curve(best_score_exact_step)
    challenge_history = load_challenge_history(CHALLENGE_FILE)

    bootstrap_steps = 0
    if loaded_transitions > 0 and loaded_games_before_csv_sync < episode_history["n_games"]:
        bootstrap_steps = min(250, max(25, len(agent.memory) // 500))
        for _ in range(bootstrap_steps):
            agent.train_long_memory()
        agent.save(
            extra_state={
                "best_score": best_score,
                "total_score": total_score,
            }
        )

    transitions_fp, transitions_writer = open_csv_writer(TRANSITIONS_FILE, transition_header)
    episodes_fp, episodes_writer = open_csv_writer(EPISODES_FILE, episode_header)
    challenge_fp, challenge_writer = open_csv_writer(CHALLENGE_FILE, challenge_header)

    step_in_episode = 0
    episode_reward = 0
    episode_step_scores = []

    print(
        f"Resume source: {load_info.get('loaded_from', 'none') if isinstance(load_info, dict) else 'none'} | "
        f"Games: {agent.n_games} | CSV transitions loaded: {loaded_transitions} | "
        f"CSV warmup steps: {bootstrap_steps}",
        flush=True,
    )

    if game.running:
        avg_score_live = (total_score / agent.n_games) if agent.n_games else 0.0
        game.set_training_info(
            {
                "game": agent.n_games + 1,
                "score": game.score,
                "best_score": best_score,
                "avg_score": f"{avg_score_live:.2f}",
                "step": step_in_episode,
                "episode_reward": episode_reward,
                "target_score": 0,
                "delta_to_target": 0,
                "epsilon": max(agent.epsilon, 0),
                "memory_size": len(agent.memory),
                "score_history": score_history[-60:],
                "ai_logs": ["AI Live Log", "Waiting for first step..."],
            }
        )

    try:
        while game.running:
            if not game.process_events():
                break

            state_old = game.get_state()
            action, decision = agent.get_action_with_debug(state_old)
            ai_logs = build_ai_logs(
                state_old,
                decision,
                head=game.head[:],
                food=game.food[:],
            )

            state_new, env_reward, done = game.step(action)

            step_in_episode += 1

            target_score = get_target_score_for_step(step_in_episode, target_curve)
            delta_to_target = game.score - target_score

            step_penalty = -0.01
            challenge_bonus = 0.0
            if delta_to_target > 0:
                challenge_bonus += 0.5
            elif step_in_episode > 25 and delta_to_target < 0:
                challenge_bonus -= 0.2

            reward = env_reward + step_penalty + challenge_bonus
            episode_reward += reward
            episode_step_scores.append((step_in_episode, game.score))

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            transition_row = [
                datetime.now().isoformat(timespec="seconds"),
                agent.n_games + 1,
                step_in_episode,
                game.score,
                action,
                reward,
                int(done),
            ] + state_to_list(state_old) + state_to_list(state_new)
            transitions_writer.writerow(transition_row)
            transitions_fp.flush()

            if done:
                episode_score = game.score
                target_at_episode_steps = get_target_score_for_step(step_in_episode, target_curve)
                delta_vs_target = episode_score - target_at_episode_steps
                efficiency = episode_score / max(1, step_in_episode)

                agent.n_games += 1
                agent.train_long_memory()

                total_score += episode_score
                best_score = max(best_score, episode_score)
                score_history.append(episode_score)
                avg_score = total_score / agent.n_games
                avg_history.append(avg_score)

                agent.save(
                    extra_state={
                        "best_score": best_score,
                        "total_score": total_score,
                    }
                )

                episodes_writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        agent.n_games,
                        episode_score,
                        best_score,
                        f"{avg_score:.2f}",
                        step_in_episode,
                        episode_reward,
                        max(agent.epsilon, 0),
                        len(agent.memory),
                    ]
                )
                episodes_fp.flush()

                challenge_writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        agent.n_games,
                        step_in_episode,
                        episode_score,
                        target_at_episode_steps,
                        delta_vs_target,
                        f"{efficiency:.4f}",
                    ]
                )
                challenge_fp.flush()
                challenge_history.append(
                    {
                        "game": agent.n_games,
                        "steps": step_in_episode,
                        "score": episode_score,
                        "target_score": target_at_episode_steps,
                        "delta": delta_vs_target,
                        "efficiency": efficiency,
                    }
                )

                for step, score_at_step in episode_step_scores:
                    if score_at_step > best_score_exact_step.get(step, -1):
                        best_score_exact_step[step] = score_at_step
                target_curve = build_target_curve(best_score_exact_step)

                print(
                    f"Game: {agent.n_games} | Score: {episode_score} | "
                    f"Best: {best_score} | Avg: {avg_score:.2f} | "
                    f"Target@{step_in_episode}: {target_at_episode_steps} | Delta: {delta_vs_target:+d}",
                    flush=True,
                )

                game.reset()
                step_in_episode = 0
                episode_reward = 0
                episode_step_scores = []

            avg_score_live = (total_score / agent.n_games) if agent.n_games else 0.0
            live_target = get_target_score_for_step(step_in_episode, target_curve)
            game.set_training_info(
                {
                    "game": agent.n_games + 1,
                    "score": game.score,
                    "best_score": best_score,
                    "avg_score": f"{avg_score_live:.2f}",
                    "step": step_in_episode,
                    "episode_reward": f"{episode_reward:.2f}",
                    "target_score": live_target,
                    "delta_to_target": game.score - live_target,
                    "epsilon": max(agent.epsilon, 0),
                    "memory_size": len(agent.memory),
                    "score_history": score_history[-60:],
                    "ai_logs": ai_logs,
                }
            )
    finally:
        transitions_fp.close()
        episodes_fp.close()
        challenge_fp.close()

        plot_path = save_training_plot(score_history, avg_history)
        step_plot_path = save_step_target_plot(target_curve, challenge_history)

        if plot_path is not None:
            print(f"Training plot saved: {plot_path}", flush=True)
        if step_plot_path is not None:
            print(f"Step-target plot saved: {step_plot_path}", flush=True)

        game.close()


if __name__ == "__main__":
    train()
