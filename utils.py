import pandas as pd
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam

from pprint import pprint

def test_model(model_fn, train_loader, test_loader, train_steps=50, val_steps=50, 
               epochs=1, iterations=5, lr=1e-4, model_params={}, save_pth=None) -> list:
    """
    Test a model for n iterations. Define the number of epochs, training steps 
    and validation steps used in each iteration.
    
    Returns the training history as a list
    """
    hists = []
    for i in range(iterations):
        model = model_fn(input_size=(256, 256, 1), output_channels=1, **model_params)
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit_generator(train_loader, steps_per_epoch=train_steps, epochs=epochs,
                                      validation_data=test_loader, validation_steps=val_steps)
        hists.append(history.history)    
        # Save trained model weights if pth given and last iteration
        if save_pth and i==iterations-1:
            model.save(save_pth)
        # Clear weights so we can retrain model from scratch in next iteration
        backend.clear_session()
    
    # Return training history
    return hists


def hists2df(hists:list):
    """
    Converts list of training histories each returned from keras.model.fit_generator
    to a pandas dataframe.
    """

    cols = list(hists[0].keys()) + ['experiment', 'epoch']
    df = pd.DataFrame(columns=cols)
    
    experiment_number = 0
    for experiment in hists:
        epoch_count = len(experiment['acc'])
        for epoch in range(epoch_count):
            r = {k: experiment[k][epoch] for k in experiment}
            r['experiment'] = experiment_number
            r['epoch'] = epoch
            df = df.append(r, ignore_index=True)
        experiment_number += 1
    return df
