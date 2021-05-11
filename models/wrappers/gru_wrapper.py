from models import LSTM_Wrapper


class GRU_Wrapper(LSTM_Wrapper):
    """ Wrapper class for GRU model """

    def __init__(self, model, optimizer, lr_scheduler, device):
        super().__init__(model, optimizer, lr_scheduler, device)

    # Everything the is same as LSTM
    # Use its methods instead

