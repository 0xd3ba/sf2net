import numpy as np
import torch
import torch.nn.functional as F
import models.base


class REINFORCE_Wrapper(models.base.BaseModel):
    """ Wrapper class for REINFORCE model """

    def __init__(self, model, optimizer, lr_scheduler, device):
        super().__init__(model, optimizer, lr_scheduler, device)

    def train(self, data, target, frame_diff=None):
        """
        Trains the model after feeding in the batch

        Input shape:  (n_frames, n_features)
        Target shape: (n_frames, )
        """

        data = data.to(self.device)
        target = target.to(self.device)
        frame_diff = frame_diff.to(self.device)

        # Reward function is
        #   0:                   When correct action is taken
        #   -1 * abs(frame_diff):  When incorrect action is taken
        #
        # Kind of different way to do things, eh
        rewards = -frame_diff  # frame_diff only contains absolute values (see utils/preprocess.py)

        action_scores = self.model(data)  # Assumes each frame is an observation
        action_probs = F.softmax(action_scores, dim=-1)  # The probability of taking each action

        # Kind of different from what we usually do in a traditional RL setting
        # Assume we took the actions by sampling from the corresponding distributions
        # for each time-step, i.e. frame

        with torch.no_grad():
            action_probs_copy = action_probs.data
            actions_took = torch.tensor([
                np.random.choice([0, 1], p=probs.numpy()) for probs in action_probs_copy
            ], dtype=torch.long)

        # Get the probabilities of the actions we took
        actions_took_probs = torch.gather(action_probs,
                                          dim=-1,
                                          index=actions_took.unsqueeze(1)).squeeze(1)

        # Set the rewards for correct actions to be 0
        actions_correct_mask = (actions_took == target)
        rewards[actions_correct_mask] = 0

        # Calculate the discounted return
        # Because the frames are finite, it can be thought of as finite episode
        # So discounting factor is 1
        # Efficient way to compute the discounted returns - Take cumulative sum in reverse order
        discounted_rewards = rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])
        log_probs = torch.log(actions_took_probs)

        loss = (log_probs * discounted_rewards).sum()
        loss = -loss  # Need to maximize the rewards, so need to do a gradient-ASCENT

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return rewards.sum().cpu().item()

    def log(self, log_dir):
        """ Saves the training logs if tensorboard monitoring is enabled """
        pass

    @torch.no_grad()
    def predict(self, data):
        """
        Predicts the frames of every audio sample as 0 or 1
            0: Frame doesn't require enhancement
            1: Frame requires enhancement
        """
        data = data.to(self.device)
        action_scores = self.model(data)                 # Shape: (n_frames, n_actions)
        action_probs = F.softmax(action_scores, dim=-1)  # Shape: (n_frames, n_actions)
        actions = action_probs.argmax(dim=-1)            # Greedily return the action with highest probability

        return actions, action_probs
