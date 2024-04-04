# Deep-learning-Challenge

### Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. The CSV contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are several columns that capture metadata about each organization.

 - **EIN** and **NAME**—Identification columns
 - **APPLICATION_TYPE**—Alphabet Soup application type
 - **AFFILIATION**—Affiliated sector of industry
 - **CLASSIFICATION**—Government organization classification
 - **USE_CASE**—Use case for funding
 - **ORGANIZATION**—Organization type
 - **STATUS**—Active status
 - **INCOME_AMT**—Income classification
 - **SPECIAL_CONSIDERATIONS**—Special considerations for application
 - **ASK_AMT**—Funding amount requested
 - **IS_SUCCESSFUL**—Was the money used effectively

### Instructions

#### Step 1: Preprocess the Data
Preprocessed the dataset using Pandas and scikit-learn.
To complete the preprocessing, followed these steps:

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
   - What variable(s) are the target(s) for your model?
   - What variable(s) are the feature(s) for your model?
2. Drop the _EIN_ and _NAME_ columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use _pd.get_dummies()_ to encode categorical variables.
7. Split the preprocessed data into a features array, _X_, and a target array, _y_. Use these arrays and the _train_test_split_ function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a _StandardScaler_ instance, fitting it to the training data, then using the _transform_ function.

#### Step 2: Compile, Train, and Evaluate the Model
Using TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

#### Step 3: Optimize the Model
Using TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If I can't achieve an accuracy higher than 75%, I'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:
* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropped more or fewer columns.
  * Created more bins for rare occurrences in columns.
  * Increased or decreasing the number of values for each bin.
  * Added more neurons to a hidden layer.
  * Added more hidden layers.
  * Used different activation functions for the hidden layers.
  * Added or reducing the number of epochs to the training regimen.
    
I have added below 3 ways to optimize the model
1. Added one more hidden layer using activation function 'relu' to the model.
2. The second attempt with the "NAME" column in the dataset.
3. In the third attempt, epochs number is increased to 150 instead of 100.

#### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.
2. **Results:** Using bulleted lists and images to support your answers, address the following questions:

 * Data Preprocessing
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the features for your model?
    - What variable(s) should be removed from the input data because they are neither targets nor features?
 
 * Compiling, Training, and Evaluating the Model
     - How many neurons, layers, and activation functions did you select for your neural network model, and why?
     - Were you able to achieve the target model performance?
     - What steps did you take in your attempts to increase model performance?
       
3. **Summary:** Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

I have added 'Analysis Report' as a .pdf file.


