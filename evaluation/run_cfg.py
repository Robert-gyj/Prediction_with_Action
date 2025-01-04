META_CONFIG = {
    'ckpt_path': "ckpt/bridge_pre_mw_ft.pt", # replace!
    'vae_path': "/home/disk2/gyj/llm/sd-vae-ft-mse/", # replace!
    'clip_path': "/home/disk2/gyj/llm/clip-vit-base-patch32/", # replace!
    'thirdview_camera': 'corner3',
    'firstview_camera': 'gripperPOV',
    'use_depth': False, # whether to use depth
    'rollout_num': 5, # number of rollouts for each tasks
    'max_steps': 20, # max planning steps for each rollout
    'video_dir': "output",
    'visualize_prediction': True,
    'denoise_steps': 50, # joint denoise step
    'task_list': ['pick-place-v2', 'basketball-v2'] # tasks you want to test
}

BRIDGE_CONFIG = {
    'ckpt_path': "ckpt/bridge_pre.pt", # replace!
    'vae_path': "/home/disk2/gyj/llm/sd-vae-ft-mse/", # replace!
    'clip_path': "/home/disk2/gyj/llm/clip-vit-base-patch32/", # replace!
    'sample_name':"sample_0"
}


INSTRUCTIONS = {
                'assembly-v2': 'assemble the object',
                'basketball-v2': 'shoot the basketball',
                'button-press-topdown-v2': 'press the button',
                'button-press-topdown-wall-v2': 'press the button',
                'button-press-v2': 'press the button',
                'button-press-wall-v2': 'press the button',
                'coffee-button-v2': 'press the button',
                'coffee-pull-v2': 'pull back cup',
                'coffee-push-v2': 'push forward cup',
                'dial-turn-v2': 'turn the dial',
                'disassemble-v2': 'disassemble the object',
                'door-close-v2': 'close the door',
                'door-open-v2': 'open the door',
                'drawer-close-v2': 'close the drawer',
                'drawer-open-v2': 'open the drawer',
                'faucet-open-v2': 'open the faucet',
                'faucet-close-v2': 'close the faucet',
                'hammer-v2': 'pick up hammer',
                'handle-press-side-v2': 'press the handle',
                'handle-press-v2': 'press the handle',
                'lever-pull-v2': 'pull the lever',
                'peg-insert-side-v2': 'insert the peg',
                'peg-unplug-side-v2': 'unplug the peg',
                'pick-out-of-hole-v2': 'pick red object',
                'pick-place-wall-v2': 'pick red object',
                'pick-place-v2': 'pick red object',
                'plate-slide-v2': 'slide forward plate',
                'plate-slide-side-v2': 'slide side plate',
                'plate-slide-back-v2': 'slide back plate',
                'plate-slide-back-side-v2': 'slide back plate',
                'soccer-v2': 'kick the soccer',
                'stick-push-v2': 'push the stick',
                'stick-pull-v2': 'pull the stick',
                'push-wall-v2': 'push the object',
                'push-v2': 'pick red object',
                'reach-wall-v2': 'reach red object',
                'reach-v2': 'reach red object',
                'shelf-place-v2': 'place blue object',
                'sweep-into-v2': 'sweep brown box',
                'sweep-v2': 'sweep brown box',
                'window-open-v2': 'open the window',
                'window-close-v2': 'close the window',
                'bin-picking-v2': 'pick green object',
                'box-close-v2': 'close the box',
                'door-lock-v2': 'lock the door',
                'door-unlock-v2': 'unlock the door',
                'hand-insert-v2': 'put box into hole',
                'handle-pull-side-v2': 'pull the handle',
                'handle-pull-v2': 'pull the handle',
                'push-back-v2': 'push back object',
                }