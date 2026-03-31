import numpy as np
import json
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

nusc = NuScenes(version='v1.0-mini', dataroot='./data', verbose=False)
helper = PredictHelper(nusc)

print("Brute-forcing target extraction...")
submission_tokens = []

for ann in nusc.sample_annotation:
    cat = ann['category_name']
    if 'human' in cat or 'cycle' in cat:
        submission_tokens.append({
            'instance_token': ann['instance_token'],
            'sample_token': ann['sample_token']
        })

print(f"Found {len(submission_tokens)} targets. Calculating trajectories...")

predictions = {}
theta = np.radians(5)
cos_t, sin_t = np.cos(theta), np.sin(theta)
R_left = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
R_right = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

for token_pair in submission_tokens:
    i_token = token_pair['instance_token']
    s_token = token_pair['sample_token']
    
    past_xy = helper.get_past_for_agent(i_token, s_token, seconds=2, in_agent_frame=False, just_xy=True)
    current_ann = helper.get_sample_annotation(i_token, s_token)
    current_xy = np.array(current_ann['translation'][:2])
    
    if len(past_xy) < 1:
        v = np.array([0.0, 0.0])
    else:
        p_t_minus_1 = np.array(past_xy[-1])
        v = current_xy - p_t_minus_1 

    traj_cv, traj_left, traj_right = [], [], []
    v_left = R_left.dot(v)
    v_right = R_right.dot(v)
    
    for k in range(1, 7):
        traj_cv.append((current_xy + k * v).tolist())
        traj_left.append((current_xy + k * v_left).tolist())
        traj_right.append((current_xy + k * v_right).tolist())
        
    predictions[f"{i_token}_{s_token}"] = {
        "predicted_trajectory": [traj_cv, traj_left, traj_right],
        "probabilities": [0.6, 0.2, 0.2]
    }

with open("cv_submission.json", "w") as f:
    json.dump(predictions, f)

print("Baseline generated. GO SUBMIT.")