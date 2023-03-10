import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

s_1 = pd.read_csv('C:/Users/mudip/OneDrive/Desktop/NETCOM DA/switchesf.csv')
r_2 = pd.read_csv('C:/Users/mudip/OneDrive/Desktop/NETCOM DA/routerf.csv')
f_3 = pd.read_csv('C:/Users/mudip/OneDrive/Desktop/NETCOM DA/Faulteventsf.csv')
g_4 = pd.read_csv('C:/Users/mudip/OneDrive/Desktop/NETCOM DA/gatewayf.csv')
p_5 = pd.read_csv('C:/Users/mudip/OneDrive/Desktop/NETCOM DA/partreplacef.csv')
v_6 = pd.read_csv('C:/Users/mudip/OneDrive/Desktop/NETCOM DA/vietnamf.csv')

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

print('\n\n')
question = input("Enter Query: ")
text=""

smart_switch_count = s_1['Machine Type'].value_counts()['Smart Switch']
vision_quadra_count = s_1['Machine Type'].value_counts()['Vision Quadra']
smart_switch_japan_count = s_1[s_1['Country Name']=='Japan']['Machine Type'].value_counts()['Smart Switch']
smart_switch_cisco_count = s_1[s_1['Delivered Site']=='Cisco']['Machine Type'].value_counts()['Smart Switch']
backend_switch_usa_count = s_1[s_1['Country Name']=='USA']['Machine Type'].value_counts()['Backend Switch']
router_b01_price = s_1[s_1['Machine Model'] == 'B01']['Price'].values[0]

text = f"There are {smart_switch_count} Smart Switch Machines. "
text += f"There are {vision_quadra_count} Vision Quadra Machines. "
text += f"There are {smart_switch_japan_count} Smart Switch Machines in Japan. "
text += f"There are {smart_switch_cisco_count} Smart Switches delivered to industry Cisco. "
text += f"There are {backend_switch_usa_count} Backend Switches delivered to USA. "
text += f"The price of Router B01 is {router_b01_price}."

#router.csv
text += " The most sold router model in India was {}.".format(r_2['Machine Model'].mode().values)

india_router_count = r_2[(r_2['Country Name'] == 'India') & (r_2['Machine Type'] == 'Router')]['Machine Type'].count()
text += " Total number of routers sold in India are {}.".format(india_router_count)

india_router_price = int(r_2[(r_2['Country Name'] == 'India') & (r_2['Machine Type'] == 'Router')]['Price'].mean())
usa_router_price = int(r_2[(r_2['Country Name'] == 'USA') & (r_2['Machine Type'] == 'Router')]['Price'].mean())
text += " The prices of routers in India and USA are {} and {}.".format(india_router_price, usa_router_price)

r_2['Delivered Date'] = pd.to_datetime(r_2['Delivered Date'])

most_sold_last_year = r_2[r_2['Delivered Date'].dt.year == 2022]['Machine Model'].mode().values
text += " Most sold machines in India last year is {}.".format(most_sold_last_year)

most_sold_last_three_years = r_2[(r_2['Delivered Date'].dt.year >= 2021) & (r_2['Delivered Date'].dt.year <= 2023)]['Machine Model'].mode().values
text += " Most sold machines in India last three years is {}.".format(most_sold_last_three_years)

most_sold_this_month = r_2[(r_2['Delivered Date'].dt.month == 3)]['Machine Model'].mode().values
text += " Most sold machines in India this month is {}.".format(most_sold_this_month)

most_sold_2022 = r_2[r_2['Delivered Date'].dt.year == 2022]['Machine Model'].mode().values
text += " Most sold machines in India 2022 is {}.".format(most_sold_2022)

most_sold_jan_2022 = r_2[(r_2['Delivered Date'].dt.year == 2022) & (r_2['Delivered Date'].dt.month == 1)]['Machine Model'].mode().values
text += " Most sold machines in Jan 2022 is {}.".format(most_sold_jan_2022)

# Convert 'Fault Date' column to datetime format
f_3['Fault Date'] = pd.to_datetime(f_3['Fault Date'])

# Get the latest fault event for machine code 555
latest_fault = f_3[f_3['Fault Type'] == 555].iloc[0]['Machine Model']
text += " The latest fault event of machine code 555 is {}.".format(latest_fault)

# Check if there is a fault for machine 567 today and if any parts were replaced
if f_3[(f_3['Fault Type'] == 567) & (f_3['Fault Date'].dt.date == pd.to_datetime('today').date())].empty == False:
    text += " Yes, there is a fault for machine 567 today."
    """
if df_3[df_3['Fault Type'] == 567]['Part Replaced'].any():
    text += " Yes, there is a part replaced for switch machine code 567."
    """

# Get the number of routers serviced today and in the last month
routers_serviced_today = f_3[f_3['Fault Date'].dt.month == 3]['Machine Model'].count()
routers_serviced_last_month = f_3[f_3['Fault Date'].dt.month == 2]['Machine Model'].count()

text += " Number of routers serviced today is {} and number of routers serviced last month is {}.".format(routers_serviced_today, routers_serviced_last_month)
# Convert the date column to datetime format
g_4['Delivered Date'] = pd.to_datetime(g_4['Delivered Date'])

# Find the unique gateway models sold in the UK
uk_gateway_models = g_4[(g_4['Country Name'] == 'UK') & (g_4['Machine Type'] == 'Gateway')]['Machine Model'].unique()

# Construct a sentence with the UK gateway models
text = f"The following gateway machine models were sold in the UK: {', '.join(uk_gateway_models)}. "

# Find the price range of gateway machines in the UK
uk_min_price = g_4[g_4['Country Name'] == "UK"]['Price'].min()
uk_max_price = g_4[g_4['Country Name'] == "UK"]['Price'].max()

# Construct a sentence with the price range of UK gateway machines
text += f"The price range of gateway machines in the UK is between {uk_min_price} and {uk_max_price}. "

# Calculate the average price of gateway machines in the UK
uk_avg_price = g_4[g_4['Country Name'] == "UK"]['Price'].mean()

# Construct a sentence with the average price of UK gateway machines
text += f"The average price of gateway machines in the UK is {uk_avg_price}. "

# Count the number of gateways delivered in the last 5 years
last_5_years_gateways = g_4[g_4['Delivered Date'].dt.year > 2018]['Machine Model'].count()

# Construct a sentence with the number of gateways delivered in the last 5 years
text += f"{last_5_years_gateways} gateway machines were delivered in the last 5 years."

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')


# Convert the 'Date' column to datetime format
p_5['Date'] = pd.to_datetime(p_5['Date'])

# Extract the part name for fault type 7890
part_replaced = p_5.loc[p_5['Fault Type'] == '7890', 'Part Name'].iloc[0]

# Extract the parts replaced for machine 7890 in the last 5 years
recent_parts_replaced = p_5.loc[p_5['Date'].dt.year > 2018, 'Part Name'].unique()

# Extract the number of parts replaced for notification update this year
num_parts_replaced = p_5.loc[p_5['Date'].dt.year == 2023, 'Part Name'].count()

# Construct the text variable using the extracted information
text = (
    f"The part replaced for the code 7890 is {part_replaced}. "
    f"The parts replaced for machine 7890 in the last 5 years were {', '.join(recent_parts_replaced)}. "
    f"The number of parts replaced for notification update this year was {num_parts_replaced}."
)

# Get unique software versions
software_versions = v_6["Software Version"].unique()
software_versions_str = ", ".join(software_versions)

# Get number of machines delivered
num_delivered = v_6["Machine Model"].count()

# Get number of VG40A machines
num_vg40a = v_6[v_6["Machine Type"] == "VG40A"]["Machine Model"].count()

# Construct the text
text = (
    f"The software version of switches in Vietnam are {software_versions_str}. "
    f"The number of machines delivered in Vietnam is {num_delivered}. "
    f"The number of VG40A machines in Vietnam is {num_vg40a}."
)

# Tokenize the input question and text
encoding = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors='pt')

# Feed the input to the model to get the start and end logits
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
start_logits, end_logits = outputs.start_logits, outputs.end_logits

# Find the answer span with the highest probability
answer_start, answer_end = torch.argmax(start_logits), torch.argmax(end_logits)
answer_span = input_ids[0][answer_start:answer_end+1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_span)
answer = tokenizer.convert_tokens_to_string(answer_tokens)

# Print the question and answer
if answer:
    print(f"Query: {question}")
    print(f"Answer: {answer}")
else:
    print("I'm sorry, I could not find an answer to your question.")

