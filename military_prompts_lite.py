# military_prompts_lite.py

def get_military_prompts():
    """
    Returns a dictionary mapping class names to concise text prompts
    for key military and civilian assets found in aerial datasets.
    """
    prompts = {
        "Person": "a photo of a person from above",
        "Car": "a photo of a car from an aerial view",
        "Bus": "a photo of a bus or a large vehicle from an aerial view",
        "Drone/UAV": "a photo of a drone or a UAV in the sky",
        "Airplane": "a photo of an airplane, a passenger jet",
        "Helicopter": "a photo of a helicopter in the air",
        "Boat": "a photo of a boat in the water",
        "Military Vehicle": "a photo of a military vehicle, like a tank or a truck, from above"
    }
    return prompts