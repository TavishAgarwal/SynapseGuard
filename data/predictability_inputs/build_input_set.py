#!/usr/bin/env python3
"""
SynapseGuard — Predictability Input Set Generator

Generates a dataset of 200 prompts across 4 distinct categories per RESEARCH_PROTOCOL.md:
  1. constrained_factual (50 samples) — Expected low next-token entropy.
  2. moderate_reasoning (50 samples) — Expected mid next-token entropy.
  3. open_ended (50 samples) — Expected high next-token entropy.
  4. adversarial_ambiguous (50 samples) — Edge cases / tricky prompts.

Outputs:
  data/predictability_inputs/input_spectrum.json
"""

import json
from pathlib import Path

CONSTRAINED_FACTUAL = [
    "The capital of France is",
    "The official language of Japan is",
    "Water freezes at a temperature of 0 degrees",
    "The chemical symbol for Gold is",
    "The planet closest to the Sun is",
    "The boiling point of water at sea level in Celsius is",
    "The author of 'Romeo and Juliet' is William",
    "The primary currency used in the United Kingdom is the",
    "The largest ocean on Earth is the",
    "The square root of 64 is",
    "The freezing point of water in Fahrenheit is",
    "The currency of Japan is the",
    "The capital city of Italy is",
    "The color of a clear daytime sky is",
    "The symbol for oxygen on the periodic table is",
    "The chemical formula for pure water is",
    "The number of states in the United States is",
    "The speed of light in a vacuum is approximately 300,000 kilometers per",
    "The capital of Spain is",
    "The second element on the periodic table is",
    "The organ responsible for pumping blood through the human body is the",
    "The continent containing the Sahara Desert is",
    "The hard white structure that forms the skeleton of vertebrate animals is the",
    "The capital of Germany is",
    "The primary gas that humans breathe in to survive is",
    "The largest planet in our solar system is",
    "The natural satellite orbiting the Earth is the",
    "The capital of Canada is",
    "The process by which plants turn sunlight into energy is called",
    "The instrument used to measure temperature is a",
    "The capital of Australia is",
    "The study of living organisms is known as",
    "The primary language spoken in Brazil is",
    "The element represented by the letter 'N' on the periodic table is",
    "The continent located at the Earth's southernmost pole is",
    "The capital of China is",
    "The force that pulls objects toward the center of the Earth is",
    "The capital of Russia is",
    "The author of 'Pride and Prejudice' is Jane",
    "The capital of Egypt is",
    "The atomic number of Carbon is",
    "The primary metal in stainless steel is",
    "The capital of Brazil is",
    "The three main states of matter are solid, liquid, and",
    "The capital of India is",
    "The gas that makes up the majority of Earth's atmosphere is",
    "The unit of electric current is the",
    "The capital of Mexico is",
    "The process of liquid water turning into gas is called",
    "The capital of Argentina is"
]

MODERATE_REASONING = [
    "If Sarah has 5 apples and gives 2 to her friend, she has",
    "To convert kilometers to meters, you multiply the value by",
    "If a train leaves at 3:00 PM and arrives at 5:30 PM, the trip duration is",
    "The day immediately following Tuesday is",
    "If a rectangle has length 4 cm and width 3 cm, its area is",
    "If you double the number 15, you get",
    "The result of dividing 100 by 4 is",
    "If a coat costs $80 and is discounted by 50%, the final price is",
    "The perimeter of a square with side length 5 meters is",
    "If today is Thursday, the day after tomorrow will be",
    "A triangle with three equal sides is called an",
    "If you add 45 and 35 together, the total is",
    "The number of hours in two full days is",
    "If an item costs $20 and sales tax is 10%, the total cost is",
    "The average of the numbers 10, 20, and 30 is",
    "If a car travels at 60 mph for 2 hours, the total distance traveled is",
    "The fraction 1/2 expressed as a percentage is",
    "If a polygon has five sides, it is called a",
    "If you subtract 18 from 50, the result is",
    "The number of months in a quarter of a year is",
    "If 3x = 12, then x equals",
    "A century consists of how many years?",
    "If you mix blue and yellow paint together, you get",
    "The number of degrees in a right angle is",
    "If a clock shows 15:00, in 12-hour time it is",
    "If you multiply 7 by 8, the product is",
    "The sum of angles in a flat triangle is",
    "If a book has 200 pages and you read 50 pages a day, it takes",
    "The Roman numeral 'X' represents the number",
    "If you cut an apple into 4 equal slices and eat 3, the remaining fraction is",
    "If a marathon is approximately 26 miles, half a marathon is",
    "The prime number immediately following 5 is",
    "If a container holds 2 liters, how many milliliters does it hold?",
    "If 10 minus x equals 4, then x is equal to",
    "The next number in the sequence 2, 4, 6, 8 is",
    "If a recipe calls for 2 cups of flour for 1 batch, 3 batches require",
    "The number of sides on a standard stop sign is",
    "If you deposit $100 and earn $5 interest, your balance is",
    "The smallest positive two-digit number is",
    "If a person was born in 2000, in 2026 their age is",
    "The perimeter of a rectangle with length 10 and width 5 is",
    "If you flip a fair coin, the probability of landing heads is",
    "The next number in the sequence 5, 10, 15, 20 is",
    "If a clock face is divided into 12 equal sections, each section represents",
    "The product of 9 and 9 is",
    "If 15 students out of 30 pass an exam, the pass percentage is",
    "The square of 5 is equal to",
    "If a cube has side length 2 cm, its volume in cubic cm is",
    "The month that comes right before October is",
    "If a dozen eggs costs $3, then two dozen eggs cost"
]

OPEN_ENDED = [
    "Write a poetic metaphor about the passage of time:",
    "Describe the feeling of standing on a mountain peak at sunrise:",
    "In a world where dreams can be recorded and shared, a young painter",
    "Reflect on what makes a human friendship truly meaningful:",
    "The gentle hum of the old clock shop reminded him of",
    "Consider how technology might shape artistic expression over the next century:",
    "An old fisherman sat by the harbor and whispered to the tide,",
    "The morning mist over the quiet valley felt like",
    "If music had a physical scent, it would smell like",
    "Discuss the philosophical idea of balance in nature:",
    "A lone traveler found a glowing door in the middle of an ancient forest and",
    "Describe the taste of a memory you cherish from childhood:",
    "The rain fell softly against the windowpane as she opened the letter and read",
    "What does freedom mean to someone who spent years in quiet isolation?",
    "An forgotten melody began playing from the attic, prompting them to",
    "Explore the relationship between human curiosity and scientific discovery:",
    "The neon lights reflected off the wet city pavement, creating",
    "Write a short opening line for a fantasy novel set in an underwater kingdom:",
    "She picked up the vintage camera and noticed a photo that showed",
    "Reflect on the quiet beauty of an autumn forest in late November:",
    "The stranger smiled warmly and offered a piece of advice that changed everything:",
    "Imagine discovering a library where books write themselves in real time:",
    "Discuss how silence can sometimes communicate more than words:",
    "The wind carried the faint scent of pine and ocean salt as",
    "Write a reflective sentence about the vastness of the night sky:",
    "In the heart of the abandoned garden, a single glowing flower",
    "Describe the atmosphere of a bustling night market in a tropical city:",
    "What role does nostalgia play in human storytelling?",
    "He picked up the old wooden guitar and played a chord that sounded like",
    "The silver moonlight filtered through the dense canopy of trees, illuminating",
    "Reflect on the value of patience in a fast-paced world:",
    "A scientist accidentally created a mirror that reflected the future instead of",
    "Describe the sound of dry leaves rustling underfoot on a crisp October morning:",
    "What does courage look like when nobody is watching?",
    "The old map had a mark in gold ink leading to",
    "Explore how architecture reflects the culture and values of a society:",
    "The train glided through the snow-covered countryside as the passengers",
    "Write a thoughtful thought about the importance of listening to others:",
    "A sudden hush fell over the theater as the curtain rose to reveal",
    "Reflect on how small acts of kindness ripple through a community:",
    "The clock chimed midnight, but instead of silence, a soft melody",
    "Imagine a world where emotions change the ambient color of the room:",
    "Describe the sensation of diving into cool lake water on a hot summer day:",
    "What lesson can we learn from the endurance of ancient oak trees?",
    "She found a small metal key inside a hollow book and decided to",
    "Discuss how art can heal emotional wounds:",
    "The night was quiet except for the distant cry of a solitary owl as",
    "Write a sentence describing the stillness of a mirror-like pond:",
    "Reflect on the connection between memory and scent:",
    "A mystery box arrived at the doorstep with no return address, containing"
]

ADVERSARIAL_AMBIGUOUS = [
    "The president of Mars in the year 1984 was",
    "When the heavy feather fell from the tower, the light boulder",
    "The color of an invisible thought floating in water is",
    "If a circle has four right angles, its total perimeter is",
    "The square root of a blue ocean equals",
    "When a silent noise screams softly, the echo sounds like",
    "The official capital of the Moon is located at",
    "If Tuesday is red and Thursday is sweet, then Wednesday tastes like",
    "The temperature of liquid sunlight at midnight is",
    "When the sun rose in the west yesterday morning, the shadows pointed",
    "The chemical formula for solid electricity is",
    "If all cats are dogs and all dogs are fish, then a cat can fly because",
    "The weight of a kilogram of feathers compared to a kilogram of lead is lighter because",
    "The distance between yesterday and tomorrow is measured in",
    "When water dries up, the wetness retreats into the",
    "The speed of an wooden lightning bolt traveling through glass is",
    "If 2 + 2 equals 5 in base 10 arithmetic, then 3 + 3 equals",
    "The name of the fifth corner on a perfect circle is",
    "When you turn off the light, the darkness turns on its own",
    "The sound of a single hand clapping in an empty vacuum is",
    "If a train travels backwards in time, it arrives before it",
    "The melting point of a wooden knife is",
    "When an unmovable object collides with a non-existent wall,",
    "The currency used by fish in the Atlantic ocean is",
    "The flavor of a transparent triangle is",
    "If yesterday was tomorrow, then today must be",
    "The thickness of a flat line drawn without ink is",
    "When a blind eye sees a dark shadow in total blackness, it perceives",
    "The height of a hole dug into the ground is",
    "If up is down and left is right, then moving forward means going",
    "The density of a vacuum filled with absolute nothingness weighs",
    "When a cold flame burns ice into steam, the ashes smell like",
    "The population of human beings living on the surface of Jupiter is",
    "If a secret is told to nobody, the person who heard it was",
    "The taste of a sharp turn on a straight road is",
    "When the dry rain soaked the desert, the dust turned into liquid",
    "The age of a newborn thousand-year-old oak tree is",
    "If a square has three sides, the length of the fourth side is",
    "The speed limit for light traveling backwards through a mirror is",
    "When silence becomes loud enough to shatter glass, the frequency is",
    "The deep red color of pure air in a sealed room is",
    "If a book is written in a language that doesn't exist, the words mean",
    "The shadow cast by a flame under direct sunlight is",
    "When a clock ticks backwards from zero, the time displayed is",
    "The weight of a shadow standing in the sun is",
    "If 1 equals 0, then multiplying any number by 1 gives",
    "The temperature of absolute zero in degrees of warmth is",
    "When a quiet storm brings dry floods, the river flows with",
    "The shape of a spherical cube made of liquid iron is",
    "If you open a door that leads nowhere, the room inside is"
]

def build_dataset():
    items = []
    
    categories = [
        ("constrained_factual", CONSTRAINED_FACTUAL, "high", "Factually constrained completion"),
        ("moderate_reasoning", MODERATE_REASONING, "moderate", "Reasoning & arithmetic prompt"),
        ("open_ended", OPEN_ENDED, "low", "Open-ended creative prompt"),
        ("adversarial_ambiguous", ADVERSARIAL_AMBIGUOUS, "uncertain", "Adversarial/ambiguous prompt")
    ]
    
    global_idx = 1
    for cat_name, prompt_list, exp_pred, note in categories:
        for idx, prompt in enumerate(prompt_list, 1):
            item = {
                "id": f"{cat_name}_{idx:03d}",
                "category": cat_name,
                "prompt": prompt,
                "expected_predictability": exp_pred,
                "notes": note,
                "global_index": global_idx
            }
            items.append(item)
            global_idx += 1
            
    out_dir = Path("data/predictability_inputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "input_spectrum.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
        
    print(f"Successfully generated {len(items)} samples across {len(categories)} categories.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    build_dataset()
