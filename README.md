# LarcAiWorkProject1

## Part 1: Exploratory Data Analysis (5 Points)

### Objective:
Interpret and understand the dataset. Bonus points will be awarded for creating interesting and insightful visualizations.

### Tasks:
- Perform a thorough exploratory data analysis (EDA) to understand the distribution and relationships of the features.
- Identify any patterns, correlations, and potential outliers in the data.
- Create visualizations to illustrate key findings and insights.

## Part 2: Binary Classification Challenge: Subscription Prediction (15 Points)

### Objective:
Predict whether a client will subscribe to a term deposit based on their demographic and past campaign contact information. Build a classification model to predict the target variable `y` (binary: "yes", "no").

### Tasks:
- **Feature Engineering:** Develop and create new features that might improve the predictive power of the model. Additional points will be granted for innovative features that enhance model accuracy.
- **Model Building:** Build a binary classification model using suitable machine learning algorithms.
- **Hyperparameter Optimization:** Perform hyperparameter tuning to improve the model's performance.
- **Model Evaluation:** Evaluate the model using the F1 Score as the primary metric.

### Evaluation Metric:
- The primary evaluation metric for the classification challenge is the **F1 Score**. This metric balances precision and recall, making it suitable for imbalanced datasets.

### Dataset Information:
- The dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable `y`).

### Features Description:
- **Numerical Features:** 
  - `age`: Age of the client.
  - `balance`: Average yearly balance in euros.
- **Categorical Features:** 
  - `job`: Type of job (e.g., "admin.", "blue-collar").
  - `marital`: Marital status (e.g., "married", "single").
  - `education`: Education level (e.g., "university.degree", "high.school").
  - `contact`: Contact communication type (e.g., "cellular", "telephone").
  - `day_of_week`: Last contact day of the week (e.g., "mon", "tue").
- **Binary Features:** 
  - `default`: Has credit in default? ("yes", "no").
  - `housing`: Has housing loan? ("yes", "no").
  - `loan`: Has personal loan? ("yes", "no").

### Deliverables:
- A detailed report on the exploratory data analysis with visualizations.
- The final classification model along with the code used for feature engineering, model building, and hyperparameter tuning.
- Documentation explaining the approach, feature engineering steps, model selection, and evaluation.
- Code to perform inference on testing data (bonus point for professionalism of functions)

By following these guidelines, participants will demonstrate their ability to interpret data, engineer features, build and optimize models, and effectively communicate their findings through visualizations and detailed reporting.
