# Cover Song Similarity

---

This projects tries to make a siamese model for cover song similarity tasks.
 - The notebooks directory has the EDA and a pipeline of how to run the project.
 - Bellow you will find more information for running and customazing the project.
 - For accessing the 2 models i trained click [me](https://drive.google.com/drive/folders/19UqPQDRTk5gPoOoW3a6_iPhHZvRiS3OR?usp=sharing).

## Preprocess the Data

To ensure the data is in the right format for training, you need to preprocess it. Follow the steps below:

### Download Data

1. **Download the dataset**  

   The data used for this project is the Da-TACOS dataset, located [here](https://github.com/MTG/da-tacos). For more information about the dataset, please refer to the repository. In this project, I use only the HCPC benchmark features.


### Cut the Data to Your Desired Length

2. **Data Cutting**  
   Depending on your model requirements, you might need to cut the data into smaller lengths. In this project i choose to cut and pad the data on a fixed length of 17000 hcpc samples per song(you can change the length accoarding to your needs).
   
   You could use any length you want but KEEP IN MIND that the current architecture of model 1 isnt capable of preprocessing files with different size features.
   You could use the adaptive model inside the src file , but it is not tested on the whole dataset, so the metrics might be off.
   
   Use the provided scripts or functions to load and cut the data into appropriate pieces.

    Resize the data:

   ```bash
    python resize_features.py --source_folder "../../data/da-tacos_benchmark_subset_hpcp" --output_folder "../../data/17K_data" --target_length 17000  
   ```

    Validate the length of your data before and after the resizing:

    ```bash
    python length_check.py --target_folder "../../data/17K_data"
    ```

### Split the Data

### 3. **Split the Data into Train/Validation/Test Sets**  

Once the data is prepared, it should be split into training, validation, and test sets. This can be done manually or by using the provided code.

For this project, I used 10K pairs(or 20K performances),the dataset is slighly smaller due to the pairing algorithm(and is slighly unbalance to the Label 1 class), the model's performance metrics were promising, so no further investigation was conducted. It is up to you to decide whether to use a smaller or larger dataset based on your computational resources and project requirements.

To create a subset of the data, you can use the following command:

Make the mapping file:

```bash
    python make_mappings.py --parent_folder "../../data/17K_data" --num_pairs 10000
    # This will create a mapping JSON file that you will need for the next code snippet.
    # Output:
    #    song_pairs.json
```

   Next, you can split the data and save the output to a target directory. You can adjust the batch size(not the batch size of the model) based on your computational resources.

   ```bash
    python split_data.py mapping_json "song_pairs.json" base_path "../../data/17K_data" output_dir "../../data/data_model" batch_size 100 sample_ratio 0.1 test_size 0.1 val_size 0.1
   ```

---

## Run the Training

### Config

#### Training Configuration:
- `epochs`: Number of training epochs (e.g., `50`).
- `batch_size`: Batch size for training (e.g., `32`).
- `learning_rate`: Learning rate for the optimizer (e.g., `0.001`).
- `early_stopping`: Number of epochs with no improvement before stopping early (e.g., `5`).

#### Data Configuration:
- `train_path_features`: Path to the training features file (e.g., `"data/data_model/train_features.h5"`).
- `val_path_features`: Path to the validation features file (e.g., `"data/data_model/val_features.h5"`).
- `test_path_features`: Path to the test features file (e.g., `"data/data_model/test_features.h5"`).
- `train_path_labels`: Path to the training labels file (e.g., `"data/data_model/train_labels.h5"`).
- `val_path_labels`: Path to the validation labels file (e.g., `"data/data_model/val_labels.h5"`).
- `test_path_labels`: Path to the test labels file (e.g., `"data/data_model/test_labels.h5"`).

#### Device Configuration:
- `device`: Choose the device for computation, options are `"cuda"` (GPU), `"mps"` (Apple Silicon), or `"cpu"` (default).


### Architecture

The model_1 architecture is `SiameseNetworkWithBatchNorm`. You can find this in the `src/model.py` file.
The model_2 architecture is `SiameseNetwork_2`. You can find this in the `src/model_2.py` file.


### Running the Training

1. **Prepare the Configuration File**  
   The configuration file `config.yml` is where you define hyperparameters such as learning rate, batch size, and number of epochs. You can modify this file to adjust the values based on your needs.

2. **Run the Training Script**  
   To start training, simply run the training script:

   ```bash
   python run.py
   ```

   The script will automatically load the configuration and model files, train the model, and log the progress(see next section).
   Additionally, the training process has an early stopping mechanism to ensure that only the best-performing model is saved, preventing overfitting and unnecessary computations.

   Note: The script is configured to run the default architecture(model_1). simply change the import and model type for the second architecture.

---
3. **Run the Test dataset**
   You cant test your trained model with the script:
   
   Note: the testing of the model is done in the previous step, this is a fast way to test more model in the test dataset.

   ```bash
    python eval_on_pretrained.py --config "src/config.yml" --model "models/best_siamese_model.pth"
   ```

## Logging

When running the training script, three folders will be created:

- **`runs/`** - Stores metadata for TensorBoard visualization.  
  You can view the training logs using TensorBoard with the following command:  

  ```bash
  tensorboard --logdir=runs
  ```

- **models/** - Contains the saved model checkpoints from your training run.
- **log/** - Includes summary statistics about the model’s performance.


## Inference the Model

Once the model has been trained, you can use it for inference.



## Config
Change the values inside the config_api.yml:
- `device`: Choose the device for computation, options are `"cuda"` (GPU), `"mps"` (Apple Silicon), or `"cpu"` (default).
- `model_path`: Path to the pre-trained model file (e.g., `"models/best_siamese_model.pth"`).


### Using the API to Call the Model

To use the trained model for inference, you can call the model through an API or directly via a function call. There are two ways to perform inference with the API: by sending two WAV files(not tested but the code exists) or by sending the HCPC features (vector). You can achieve this by either using Docker or by spinning up a local server.

#### 1. **Running the API Localy**  

To run locally:

```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 
    #spins up the server
```

This will start a local server with 4 worker processes, and the inference API will be available at `http://localhost:8000`.

You can test the model:
```bash
    python test_api.py
    # dive deeper in to the file to select your songs
```

#### 2. **Running the API on Docker**  
You can run the inference API by using Docker.

- **Using Docker**  
  To use Docker, ensure you have the Docker image built and then run the following command:

```bash
docker build --build-arg MODEL_PATH= <path_to_model> --build-arg SRC_PATH=src -t <name_of_your_image> .   
```


  ```bash
  docker run -p 8000:8000 <name_of_your_image>
  ```

  This will spin up a Docker container with the inference API running on port 8000.

You can test the model:
```bash
    python test_api.py
    # dive deeper in to the file to select your songs
```

### Run the test

You can run the test with pytest:

```bash
    pytest <test.py>
```

---

### **Note:**  
- Make sure to keep the `config.yml` file and model weights files in the proper directories.
- Ensure your environment is correctly set up for all dependencies to run smoothly.

---