import pandas as pd
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

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
        # compile model
        model = model_fn(output_channels=1, **model_params)
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        
        # Reduce learning rate on plateau. Patience is one less than early stopping to give
        # new learning rate a try
        patience = 1
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=patience-1, verbose=1)
        
        # Stop training when validation accuracy decreases on subsequent epoch
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience+1, verbose=1)
        callbacks = [reduce_lr, early_stopping]
        
        # Save best model if path provided
        if save_pth:
            save_model = ModelCheckpoint(save_pth, save_best_only=True, monitor='val_acc')
            callbacks += [save_model]
        
        # Fit the model
        history = model.fit_generator(train_loader, steps_per_epoch=train_steps, epochs=epochs,
                                      validation_data=test_loader, validation_steps=val_steps,
                                      callbacks=callbacks)
        hists.append(history.history)    

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
