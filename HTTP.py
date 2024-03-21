import requests

# Define the URL and JSON data
url = 'http://<ESP_IP_ADDRESS>/openDoor'
json_data = {
    "action": "openDoor",
    "parameters": {
        "robotId": "TurtleBot3_ID"
    }
}
message = ''
def HTTP_requests():
# Send the POST request
    response = requests.post(url, json=json_data)

    # Check the response
    if response.status_code == 200:
        # Request was successful
        response_data = response.json()
        if response_data["status"] == "success":
            global message
            message = response_data["data"]["message"]
            print("Success:", message)
        else:
            print("Request failed with message:", response_data["data"]["message"])
            message = HTTP_requests()
    else:
        # Request failed
        print("Request failed with status code:", response.status_code)
        message = HTTP_requests()
    return message