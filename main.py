import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pyautogui
import pygetwindow as gw
from ultralytics import YOLO
import cv2
from PIL import ImageGrab, Image
import easyocr
import autogen
from IPython import get_ipython
import time
from urllib.parse import urlparse
from transformers import AutoModelForCausalLM, AutoTokenizer

#from autogen import GroupChatManager, GroupChat
# from sort.sort import sort
# import numpy as np

# ele_tracker = sort.Sort()
# Run a default cuda device
# torch.set_default_device("cuda")

# Check if CUDA is available and use it if so
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used:",device)
# Set torch device
torch.set_default_device(device)
#Create imp vision model SMVM model
model = AutoModelForCausalLM.from_pretrained(
"MILVLG/imp-v1-3b", 
torch_dtype=torch.float16, 
device_map="auto",
trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)
# Create OCR model
reader = easyocr.Reader(['en'], gpu=True)
# YOLO model 'best.pt' classes
class_names = ['Button', 'Edit Text', 'Header Bar', 'Image Button', 'Image View', 'Text Button', 'Text View']
# Define dictionary to map class IDs to class names
class_id_to_name = {i: class_name for i, class_name in enumerate(class_names)}
# Create the YOLO model
modely = YOLO('best.pt')

# Configure Chrome options with the temporary user data directory
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument("user-data-dir=C:\\Users\\User\\AppData\\Local\\Google\\Chrome\\User Data")
# Initialize the WebDriver with Chrome options
# driver = webdriver.Chrome(options=chrome_options)

# Initialize webdriver with default chrome withou user logging to chrome
driver = webdriver.Chrome()
# Set the window size (replace 'width' and 'height' with your desired values)
driver.set_window_size(480, 800)
# Set the window position (replace 'x' with the horizontal position you want)
driver.set_window_position(0, 0)  # For example, position it to the right side of the screen
# Navigate to Google and perform a search
google_url = "https://www.google.com"
driver.get(google_url)


# Capture window screenshot
def capture_window_screenshot():
    
    time.sleep(1)       
    # Get the current window handle
    # swindow_title = driver.current_window_handle
    swindow_title = driver.title   
    window = gw.getWindowsWithTitle(swindow_title)
    if len(window) == 0:
        print(f"Window with title '{swindow_title}' not found.")
        return
    time.sleep(1)
    
    # print('00window', window[0])
    # window[0].activate()
    # Use pyautogui to activate the Chrome window by clicking its title in the taskbar
    #pyautogui.click(swindow_title)
    # print('11window', window[0])

    # get the x,y coordinates of the window
    window = window[0]
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    # Capture a screenshot of the window
    screenshot = ImageGrab.grab(bbox=(left, top+150, right, bottom))    
    # Save the screenshot to a file
    screenshot.save("./image/wss.png")
    time.sleep(1)

# Detect and return the element in the window
def screeninfo():
    
    # Load the image using OpenCV
    image_path = "./image/wss.png"
    img = cv2.imread(image_path)
    # Run YOLO object detection and get the detections
    detections = modely(image_path)[0]

    # Draw bounding boxes on the image
    for det in detections.boxes.data.tolist():
        xmin, ymin, xmax, ymax, conf, cls = det
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        label = f'{modely.names[int(cls)]} {conf:.2f}'
        color = (0, 255, 0)  # Green color for the bounding box
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        img = cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes to a file
    cv2.imwrite('detected_image.jpg', img)
    # Show the image with bounding boxes
    #cv2.imshow('Image with Bounding Boxes', img)

    screenshot_array = cv2.imread(image_path)

    # screenshot_array = np.array("./trial/image/wss.png")
    detections_ = []
    extracted_text_list = []

    
    for detection in detections.boxes.data.tolist():
        
        x1, y1, x2, y2, score, class_id = detection
        # scorer = round(score, 2)      
        # Get the class name using the class ID
        class_name = class_id_to_name.get(class_id, 'Unknown')  # Default to 'Unknown' if class ID is not found
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Crop the screenshot to the size of each element
        element_crop = screenshot_array[ y1 : y2, x1 : x2, :]
        # OCR the text within each element
        tdetections = reader.readtext(element_crop)
        
        merged_text = ""
        extracted_text_list = []

        # Join OCR for each element to the element class and location
        for tdetection in tdetections:
            bbbox, text, score = tdetection

            #text = text.replace(' ', '')

            # Append the extracted text to the list
            extracted_text_list.append(text)
       
        # Merge the extracted text into a single string
        merged_text = ' '.join(extracted_text_list)
    
        print ('extracted text:', merged_text)

        # Get the center coordinates for each element 
        x = int(x1+(x2-x1)/2)       
        y = int(y1+(y2-y1)/2)+150

        # Append a list for all detected elements complete information
        detections_.append([x, y, class_name, merged_text])

    # print the list of complete elements detected wih their information
    print(detections_)

    # Sort the elements detected asccending based on their y coordiante value 
    detections_.sort(key=custom_sort)
    # Filter out cells with empty merged_text
    filtered_detections = [detection for detection in detections_ if detection[-1] != ""]

    # Get the title of the opened window
    window_title = driver.title    
    # insert window title to the list    
    filtered_detections.insert(0, f'Window title is:{window_title}')
    # Get the current coordinate of the mouse
    m, p = pyautogui.position()
    # Inser the current coordinate of mouse to the list
    filtered_detections.insert(0, f'Mouse Position: x={m}, y={p}')
    
    # Index the sorted detections possible give LLM a id for each element instead of location..
    # indexed_sorted_detections = list(enumerate(filtered_detections))

    return filtered_detections

# Describe the window using vision model
def showscreen(show):
     
    # Wait for the page to load completely
    wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds in this example)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

    # capture sceenshot for window
    capture_window_screenshot()
      
    # Create a tokenizer for vision model
    # tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)
    # tokenizer.to(device)

    #Set inputs for imp vision model
    question = show
    text = f"A chat between a blind user who can't see the screen and you the artificial intelligence assistant with vision. As the assistant provide in details the important content information inside window screenshot include main buttons for possible interaction and answer their specific questions. USER: <image>\n{question} ASSISTANT:"
    image = Image.open("./image/wss.png")
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    image_tensor = model.image_preprocess(image).to(device)

    #Generate the answer
    output_ids = model.generate(
    input_ids,
    max_new_tokens=2000,
    images=image_tensor,
    use_cache=True)[0]
    
    # Vision model text response
    img2txt = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()      
    # screen_show = f'Chrome window shows {img2txt}. Here is elements of the with x,y locations and object identification/OCR elements:\n {detections_} \n "Please respond based on the screen information provided above (including OCR results and extracted location)' 

    return img2txt

# Get the elements information
def getelements(order):
    
    # Wait for the page to load completely
    wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

    # capture window
    capture_window_screenshot() 
    # get the elements
    detections_ = screeninfo()
        
    screen_elements = f'Here are the elements inside the webpage with x,y locations and object identification/OCR content of elements:\n {detections_}' 

    return screen_elements

# Search google or enter a url and retur the window description and elements to LLM
def gaddressbar(url_or_search_term):

    # Navigate to a URL by typing in the address bar
    input_text = url_or_search_term
    # Check if the input is a valid URL
    parsed_url = urlparse(input_text) 

    if parsed_url.scheme and parsed_url.netloc:
        # It's a valid URL, so open it with driver.get                   
        driver.get(input_text)
        # Wait for the page to load completely
        wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds in this example)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

    else:

        # Navigate to Google and perform a search
        google_url = "https://www.google.com"
        driver.get(google_url)      
        # Wait for the page to load completely
        wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds in this example)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[name="q"]')))
        search_box = driver.find_element(By.CSS_SELECTOR, '[name="q"]')  # Locate the search box by CSS selector
        time.sleep(1)
        search_box.send_keys(input_text)  # Type the input (URL or search term)
        time.sleep(1)
        search_box.send_keys(Keys.RETURN)  # Press Enter to initiate the search
        wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds in this example)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
         
    # Send to vision model to describe the window
    msg2vision = f'{url_or_search_term}, extract main results shown on the webpage'
    screen_show = showscreen(msg2vision)
    # Screen_show = showscreen("Describe and write information data results shown in the webpage?")
    # Send to elements function to get the elements
    screen_elements = getelements("order")


    return screen_show, screen_elements

# Open Chrome window and set its diminsions and coordinates
def ochrome():

    # Maximize the browser window
    #driver.maximize_window()
    # Set the window size (replace 'width' and 'height' with your desired values)
    driver.set_window_size(480, 800)
    # Set the window position (replace 'x' with the horizontal position you want)
    driver.set_window_position(0, 0)  # For example, position it to the right side of the screen  
    # Wait for the page to load completely
    wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds in this example)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

# Define a custom sorting function to sort by y1 coordinate for elements in asscending order
def custom_sort(detection):
    return detection[1]  # Sort based on the y1 coordinate

# execute written code by LLM
def execute(language, code):   

    # execute a code based on language
    result = coderexe.execute_code_blocks([(language, code)])
    
    # get screen info after executing the code
    screen_show = showscreen("What information shown in the webpage?")
    screen_elements = getelements("order")
    
    # return coderexe.execute_code_blocks([("sh", script)])
    # messages = [{"content":code}]
    
    return result, f"this is the updated page after code execution {screen_show}, and these are the updated elements of screen {screen_elements}"

# Enter python code to execute
def exec_python(cell):
    try:
        exec(cell, globals(), locals())
        time.sleep(2)
        # Wait for the page to load completely
        wait = WebDriverWait(driver, 10)  # Adjust the timeout (10 seconds in this example)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        return {
            "result": "Code executed successfully.",
            "log": locals() if isinstance(locals(), dict) else None
        }
    except Exception as e:
        return {
            "result": f"Error: {str(e)}",
            "log": None
        }

# LLM Model and API
config_list = [
    {
        "model": "gpt-3.5-turbo-1106",
        "api_key": "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"
    }
]
llm_config={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,    
}

# Agent Planner setting
planner = autogen.AssistantAgent(
    
    name="planner",
    
    human_input_mode="NEVER",
    
    llm_config={
        "seed": 42,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0.5,  # temperature for sampling
    
        "functions": [
            {
                "name": "auto_execute_code",
                "description": 
                "Use this function to reach your goal by writing code using your coding and language skills. you can write python code (in a python coding block) or shell script (in a sh coding block) for the proxy system to execute it for you.", 
                                
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "description": "language script type, <sh> for shell script, or <python> for python code"
                        },
                        "code": {
                            "type": "string",
                            "description": "The user can't modify your output. So do not suggest incomplete code which requires user to modify it. Don't write a code block if it's not intended to be executed by the proxy system. you can write shell command script, you can write Python code, for examples as: 1. When you need to interact with chrome window elements use python pyautogui code and element center coordiantes given to you with chrome window elements list to click on buttons/fields/type, make mouse and keyboard type inputs with 0.3 second duration for inputs to simulate human behavior, write multisteps code. 2. When you need to collect info, write operational code to output the info you need, for example, download/read/create a file, print the content of a webpage or a file, draw, get the current date/time, check the operating system, use an app. After sufficient info is printed and the task is ready to be solved, solve the task by yourself. 3. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.",
                        },                   
                    },
                    "required": ["language", "code"],
                },
            },
            {
                "name": "gaddressbar",
                "description": "Use this function to navigate to a URL, enter complete URL starting with https:// or for google search to be entered in google chrome address bar ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url_or_search_term": {
                            "type": "string",
                            "description": "Write only the <URL> or the search term",
                        }
                    },
                    "required": ["url_or_search_term"],
                },
            },           
           
            {
                "name": "showscreen",
                "description": "Use this function to request any information about what is shown in the screen, it returns content information about what is shown in the chrome window to help you collect information and possible functions/buttons in the page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "show": {
                            "type": "string",
                            "description": "ask detailed question what content and information are you looking for in the page",
                        }
                    },
                    "required": ["show"],
                },
            },

            {
                "name": "getelements",
                "description": "Use this function to get elements locations so you can then use them to write python pyautogui lib script to interact with the page click/write etc, view elements of webpage as a list of elements with their x,y locations and content, so you can interact with elements as needed write, -get- to get the elements list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order": {
                            "type": "string",
                            "description": "-get-",
                        }
                    },
                    "required": ["order"],
                },
            },
        
        ],

    
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    
    
    system_message=  

            """
                You are a very helpful assistant for a blind person who cant see the computer screen, you will operate chrome instead of user to perform various types of activities              
                You can use the provided functions call to help you code and execute code for you and user and feedback result, and to help you navigate URLs and to help you search google which will open a chrome window and search or enter the URL in chrome window. 
                Use mouse keyboard code input to interact with chrome window elements 
                chrome window list of object detection of elements on chrome window [h, v, text found in element based on OCR], window size 480x800, the results of elements info can be incomplete, interpretate elements based on content/location to understand their functionality and information in them so to operate web, Iterate again and use a different approach if you failed to achieve what you want.
                You can ask for information about what is in the page by using the showscreen function and ask for what are you looking for in the screen.
                Solve the tasks step by step to accomplish a project if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
                If you want to save a file in the system directory before executing for your project, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant or use file output. Check the execution result returned by the system.
                If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need using search/browse/showscreen/getelements functions, and think of a different approach to try.
                When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

                Reply TERMINATE when the task is completed.
            """


)

# user proxy agent settings
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=7,
    # is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    # function_map={"showscreen": showscreen(),
    #                "gaddressbar": gaddressbar()},
    code_execution_config={
        
        "work_dir": "coding",
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
)

# An agent for executing code
coderexe = autogen.UserProxyAgent(
    name="coderexe",
    human_input_mode="ALWAYS",
    # is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=0,
    code_execution_config={
        
        "work_dir": "coding",
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
)
    
# register the functions to user_proxy agent
user_proxy.register_function(
    function_map={
        "showscreen": showscreen,
        "gaddressbar": gaddressbar,
        "getelements": getelements,
        "auto_execute_code": execute,
    }
)

# message = f'Based on the following information of shown webpage with its elements coordinates, write a python code with pyautgui that will write the name `Husam` inside the element Type Name, then click the Enter button, use middle element click based the required coordinates provided in information: {detections_}'
# "what is the weather currently in amman?"
# "write code to scrape and plot the yearly deaths from car accendents in USA for the last 10 years from wikipedia data"
# "go to autodraw.com, open the drawing area then draw a complete flower in it, user pyautogui for that" 

# Enter first question or leave empty
message = ""
# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(   
    planner,
    message=message
    #message="""What are the main political news in the Middle East today?""",
)

input("Press Enter to close the browser...")

# unused agent
csender = autogen.AssistantAgent(
    
    name="planner",
    
    human_input_mode="NEVER",
    
    llm_config={
        "seed": 42,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0.5,  # temperature for sampling
        
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    
)

# unused agent
codersender = autogen.UserProxyAgent(
    name="codersender",
    human_input_mode="NEVER",
    
    # is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)








