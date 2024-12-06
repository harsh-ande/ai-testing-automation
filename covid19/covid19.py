import pandas as pd
import openai

# Set your OpenAI API key
openai.api_key = "<redacted>"

def chatgpt_predict(input_data):
    """
    Function to interact with ChatGPT using the chat/completions endpoint and get predictions.
    """
    messages = [
        {"role": "system", "content": "You need to predict based on the given conditions, whether someone has COVID-19 or not."},
        {"role": "user",
         "content": f"Based on the following details, does the person have COVID-19? Answer with 'YES' or 'NO'. "
                    f"In the input table, 'infectionProb' is the target label where 1 indicates 'YES' (COVID positive) "
                    f"and 0 indicates 'NO' (COVID negative). \nInput table: {input_data}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Specify the model explicitly
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    return response['choices'][0]['message']['content'].strip().upper()

# Load your dataset
file_path = "covid19data.csv"
covid_data = pd.read_csv(file_path)

results = []

# Process each row in the dataset
for idx, row in covid_data.iterrows():
    input_data = row.drop(labels=["infectionProb"]).to_dict()
    expected_output = "YES" if row["infectionProb"] == 1 else "NO"
    prediction = chatgpt_predict(input_data)
    result = "Pass" if prediction == expected_output else "Fail"
    print(f"Test Case {idx + 1}: Expected = {expected_output}, Predicted = {prediction}, Result = {result}")
    results.append({
        "Test_Case_ID": idx + 1,
        "Input_Data": input_data,
        "Expected_Output": expected_output,
        "Predicted_Output": prediction,
        "Result": result
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("chatgpt_covid19_test_results.csv", index=False)
print("Test results saved to 'chatgpt_covid19_test_results.csv'")
