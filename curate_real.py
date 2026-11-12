import json
import os
import asyncio
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import edge_tts

# Let's curate 30 top-quality real Indian-English audio samples from our downloaded Svarah set
# (20 for train, 10 for eval) with clean, grammatical transcripts.
real_svarah_curated = [
    # 20 for Train
    ("svarah_001.wav", "Give me a list of the contacts I sent money to last week.", "fintech_upi", "train"),
    ("svarah_003.wav", "Set a reminder for my pending payment to contact Mahnoor.", "fintech_upi", "train"),
    ("svarah_005.wav", "Whenever a payment got stuck they used to refund it within a day.", "fintech_upi", "train"),
    ("svarah_006.wav", "Can you add my India Post Payments Bank Mastercard?", "fintech_upi", "train"),
    ("svarah_007.wav", "Set a reminder for my pending payment to contact Sumaya.", "fintech_upi", "train"),
    ("svarah_011.wav", "Do you store personal information about me?", "software_engineering", "train"),
    ("svarah_014.wav", "My money got deducted but my order of Cocosutra hot chocolate is not confirmed.", "food_delivery", "train"),
    ("svarah_015.wav", "Turn off notifications for offers and rewards in Axis Bank Lime.", "fintech_upi", "train"),
    ("svarah_016.wav", "Where should I go for Mexican food in my area?", "food_delivery", "train"),
    ("svarah_019.wav", "Can I know the nearest medical store to my current location in district Dindigul?", "medical_counselling", "train"),
    ("svarah_021.wav", "Can you send money to unsaved mobile numbers in Pay?", "fintech_upi", "train"),
    ("svarah_027.wav", "I am in a meeting, please do not speak.", "daily_workplace", "train"),
    ("svarah_028.wav", "Did I receive an e-copy of the bill from the last transaction to FirstCry?", "fintech_upi", "train"),
    ("svarah_029.wav", "Disable autopay for my broadband.", "fintech_upi", "train"),
    ("svarah_030.wav", "Can I take a photo of my card to add it to PayPal?", "fintech_upi", "train"),
    ("svarah_031.wav", "Choose district to search for medical stores in state Chhattisgarh.", "medical_counselling", "train"),
    ("svarah_032.wav", "Create a new reminder for paying the gas bill.", "fintech_upi", "train"),
    ("svarah_033.wav", "Report unauthorized MobiKwik account activity.", "fintech_upi", "train"),
    ("svarah_034.wav", "Block the ONGC holdings from my account.", "fintech_upi", "train"),
    ("svarah_036.wav", "Will my credit score be checked for initiating a personal loan?", "fintech_upi", "train"),
    # 10 for Eval
    ("svarah_039.wav", "I want to order chocolates, can you show me what all is available in that category?", "food_delivery", "eval"),
    ("svarah_041.wav", "Are there going to be any discounts on vegan items in this week?", "food_delivery", "eval"),
    ("svarah_043.wav", "Show the products in bakery with high ratings.", "food_delivery", "eval"),
    ("svarah_045.wav", "Look online for any regulations you need to follow.", "daily_workplace", "eval"),
    ("svarah_051.wav", "Give me the list of contacts who I have received money from in the last month on all payment apps.", "fintech_upi", "eval"),
    ("svarah_055.wav", "Click the line below the header that says business name and type the name of your business.", "software_engineering", "eval"),
    ("svarah_060.wav", "What is the balance in my Tamil Nadu Grama Bank account?", "fintech_upi", "eval"),
    ("svarah_062.wav", "Delete the reminder for education loan.", "fintech_upi", "eval"),
    ("svarah_064.wav", "I am unable to add a bank account to my PayZapp.", "fintech_upi", "eval"),
    ("svarah_067.wav", "My Paytm reward has not been credited to my bank account.", "fintech_upi", "eval"),
]

print("Curated real samples:", len(real_svarah_curated))
