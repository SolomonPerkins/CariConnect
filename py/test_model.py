import requests
import json

def test_recommendation():
    url = 'http://localhost:8000/recommend'
    
    # Test data
    data = {
        'title': 'Machine Learning Basics',
        'subjects': 'Computer Science',
        'synopsis': 'An introduction to fundamental machine learning concepts and algorithms.'
    }
    
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    try:
        # Make request
        print("Sending request to:", url)
        print("With data:", json.dumps(data, indent=2))
        print("Headers:", headers)
        
        response = requests.post(
            url,
            json=data,
            headers=headers,
            verify=False  # Only if needed for local testing
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            print(f"Response Text: {response.text}")
        except:
            print("Could not print response text")
        
        if response.status_code == 200:
            recommendations = response.json()
            print("\nRecommendations:")
            # for i, rec in enumerate(recommendations['recommendations'], 1):
            #     print(f"\nRecommendation {i}:")
            #     print(f"Title: {rec['title']}")
            #     print(f"Subjects: {rec['subjects']}")
            #     print(f"Publisher: {rec['publisher']}")
            #     print(f"Similarity Score: {rec['similarity_score']:.4f}")
            print("HERE: ", recommendations)
        else:
            print(f"\nError Response:")
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is the Flask app running?")
    except json.JSONDecodeError:
        print("Error: Could not decode the response JSON")
    except Exception as e:
        print(f"Error here: {str(e)}")

if __name__ == '__main__':
    test_recommendation()
