import pandas as pd
import openai

# Set your OpenAI API key
openai.api_key = "<redacted>"


def chatgpt_predict(input_data):
    """
    Function to interact with ChatGPT using the chat/completions endpoint and get predictions.
    """
    # Prepare the input prompt for ChatGPT
    messages = [
        {"role": "system", "content": "You need to predict based on the given conditions, whether someone has lung cancer or not."},
        {"role": "user",
         "content": f"Based on the following details, does the person have lung cancer? Answer with 'YES' or 'NO'. In the input table, if the value of a cell is 2 it means YES and if its 1 it means NO. \nInput table: {input_data}"}
    ]

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use 'gpt-4' or another chat-based model if needed
        messages=messages,
        max_tokens=10,  # Keep response short
        temperature=0  # Ensures deterministic answers
    )

    # Extract and clean the response
    answer = response['choices'][0]['message']['content'].strip().upper()
    return answer


# Load the test data from the provided CSV
test_data = pd.read_csv('./survey_lung_cancer.csv')

# Prepare a list to store results
results = []

# Iterate through each row in the test data
for _, row in test_data.iterrows():
    # Prepare input for the chatbot
    input_data = row.drop(labels=["LUNG_CANCER"]).to_dict()
    expected_output = row["LUNG_CANCER"].upper()

    # Call ChatGPT and get the prediction
    prediction = chatgpt_predict(input_data)

    # Print the expected and predicted outputs
    print(f"Test Case {_ + 1}: Expected Output = {expected_output}, Predicted Output = {prediction}")

    # Record the result
    results.append({
        "Test_Case_ID": _ + 1,  # Index as Test Case ID
        "Input_Data": input_data,
        "Expected_Output": expected_output,
        "Predicted_Output": prediction,
        "Result": "Pass" if prediction == expected_output else "Fail"
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
output_csv_path = "./chatgpt_predictions.csv"
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
summary_file_path = "./chatgpt_test_summary.txt"
with open(summary_file_path, "w") as summary_file:
    summary_file.write(summary)

print(f"Summary saved to {summary_file_path}")
