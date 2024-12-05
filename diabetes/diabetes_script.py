import pandas as pd
import openai

# Set your OpenAI API key
openai.api_key = "<redacted>"


def chatgpt_predict(input_data):
    """
    Function to interact with ChatGPT using the chat/completions endpoint and get predictions.
    """
    messages = [
        {"role": "system", "content": "You are an AI model tasked with predicting whether a person has diabetes based on given health parameters."},
        {"role": "user",
         "content": f"Based on the following health details, does the person have diabetes? Just return one word answer with 'YES' or 'NO'.\n\nDetails: {input_data}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=messages,
        max_tokens=10,  
        temperature=0  
    )

    answer = response['choices'][0]['message']['content'].strip().upper()
    return answer


# Load the diabetes test data
test_data = pd.read_csv('survey_diabetes.csv')

# Prepare a list to store results
results = []

# Iterate through each row in the test data
for idx, row in test_data.iterrows():
    # Prepare input for the chatbot by excluding the 'Diabetes' column
    input_data = row.drop(labels=["Diabetes"]).to_dict()
    expected_output = row["Diabetes"].upper()

    # Call ChatGPT and get the prediction
    prediction = chatgpt_predict(input_data)

    # Print the expected and predicted outputs
    print(f"Test Case {idx + 1}: Expected Output = {expected_output}, Predicted Output = {prediction}")

    # Record the result
    results.append({
        "Test_Case_ID": idx + 1,  # Index as Test Case ID
        "Input_Data": input_data,
        "Expected_Output": expected_output,
        "Predicted_Output": prediction,
        "Result": "Pass" if prediction == expected_output else "Fail"
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
output_csv_path = "./diabetes_predictions.csv"
results_df.to_csv(output_csv_path, index=False)

# Generate Summary and Statistics
total_tests = len(results_df)
total_pass = len(results_df[results_df["Result"] == "Pass"])
total_fail = len(results_df[results_df["Result"] == "Fail"])
pass_rate = (total_pass / total_tests) * 100
fail_rate = (total_fail / total_tests) * 100

summary = f"""
Automation Test Summary:
------------------------
Total Test Cases: {total_tests}
Passed: {total_pass}
Failed: {total_fail}
Pass Rate: {pass_rate:.2f}%
Fail Rate: {fail_rate:.2f}%
"""

# Print the summary
print(summary)

# Save the summary to a text file
summary_file_path = "./diabetes_test_summary.txt"
with open(summary_file_path, "w") as summary_file:
    summary_file.write(summary)

print(f"Summary saved to {summary_file_path}")
