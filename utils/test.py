import numpy as np

# define a function to test in all the data
def prediction_error_compute(dataset_test, rollout_model_list):
    prediction_error = []
    obs_name = ['object', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', \
               'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', \
                'robot0_gripper_qvel']

    for case_i in range(len(dataset_test)):
        tmp_ob_dict = {}
        for key in obs_name:
            if key in dataset_test[case_i]['obs'].keys():
                tmp_ob_dict[key] = dataset_test[case_i]['obs'][key]

        tmp_prediction_list = []
        for model_id in range(len(rollout_model_list)):
            tmp_prediction_list.append(rollout_model_list[model_id](ob=tmp_ob_dict, goal=None))

        tmp_prediction_list = np.array(tmp_prediction_list)
        prediction_error.append(dataset_test[case_i]['actions'] - tmp_prediction_list.var(axis = 0))

    return prediction_error