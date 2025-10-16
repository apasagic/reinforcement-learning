import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ----- Setup Video Saving -----
video_filename = "agent_run.mp4"
frame_size = (84, 84)
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

# ----- Setup Live Plot Window -----
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((84, 84, 3), dtype=np.uint8))  # RGB image
ax.set_title("Agent View")

# ----- Environment Reset -----
state = env.reset()[0]
state = np.array(state, dtype=np.float32) / 255.0
state = np.transpose(state, (1, 2, 0))  # HWC format

num_steps = 500
done = False

for step in range(num_steps):
    # --- Choose action ---
    action = DQN_agent.get_action(state, env)

    # --- Environment step ---
    state_new, reward, done, truncated, info = env.step(action)
    state_new = np.array(state_new, dtype=np.float32) / 255.0
    state_new = np.transpose(state_new, (1, 2, 0))  # HWC format
    state = state_new  # update state

    Q_value = np.mean(DQN_agent.QNetwork.predict(state.reshape(1,84,84,4),verbose=0))

    if "episode" in info:
        ep_info = info["episode"]   
        print(f"Episode {ind_episode + 1}: reward = {ep_info['r']}, length = {ep_info['l']}, epsilon = {DQN_agent.epsilon}")
        break

    #text = f"Action: {action},Reward: {ep_info['r']}"
    text = f"QVal: {Q_value:.3f}, Action: {action}"

    # --- Extract grayscale frame for display ---
    frame = state[..., -1]  # Use latest frame in stack
    frame_uint8 = (frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

    # --- Add action label to upper-right ---
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    color = (255, 255, 255)
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = frame_bgr.shape[1] - text_width - 2
    y = text_height + 2
    cv2.putText(frame_bgr, text, (x, y), font, font_scale, color, thickness)

    # --- Save frame to video ---
    video_writer.write(frame_bgr)

    # --- Display live animation ---
    im.set_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1 / fps)

    if done or truncated:
        state = env.reset()[0]
        state = np.array(state, dtype=np.float32) / 255.0
        state = np.transpose(state, (1, 2, 0))

# ---- Cleanup ----
video_writer.release()
env.close()
plt.ioff()
print(f"Video saved to: {video_filename}")
