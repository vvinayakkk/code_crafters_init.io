from flask import Flask, request, jsonify
from flask_cors import CORS
from twilio.rest import Client
import os
import time
from langchain.agents import AgentType, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from composio_langchain import ComposioToolSet
from docx import Document
import markdown
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Twilio configuration
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')# Twilio Sandbox Number
ACCOUNT_SID = os.getenv('ACCOUNT_SID')
AUTH_TOKEN = os.getenv('AUTH_TOKEN')
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Initialize LangChain components
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
def initialize_agent_with_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # Initialize Composio tools
    composio_toolset = ComposioToolSet(api_key="irwfwjxhkxng4167sj48")
    tools = composio_toolset.get_tools(actions=['GMAIL_SEND_EMAIL','GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN','GOOGLEDOCS_GET_DOCUMENT_BY_ID'])
    
    # Create agent
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# Initialize the agent
agent = initialize_agent_with_tools()

def send_whatsapp_alert(to_number, message):
    """Send WhatsApp notification using Twilio"""
    try:
        message = client.messages.create(
            body=message,  # Corrected parameter
            to=f'whatsapp:{to_number}',
            from_=TWILIO_WHATSAPP_NUMBER,  # Added from_ parameter
        )
        print(f"Twilio response: {message.sid}")
        return {"success": True, "message_sid": message.sid}
    except Exception as e:
        print(f"Twilio error: {str(e)}")
        return {"success": False, "error": str(e)}

def send_email_alert(recipient, subject, content):
    """Send email notification using Composio through LangChain agent"""
    try:
        result = agent.run(
            f"Send an email to {recipient} with the subject '{subject}' and the following content: {content}"
        )
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

def evaluate_security_threat(model_analysis, gemini_analysis):
    """Evaluate the security threat level based on model analysis and Gemini's interpretation"""
    threat_detected = False
    severity = "Low"
    details = ""
    
    # Extract top prediction and confidence
    top_class = model_analysis.get("top_classes", [""])[0]
    top_confidence = model_analysis.get("confidences", [0])[0]
    
    # Determine if there's a threat based on model classification
    security_threats = ["Burglary", "Robbery", "Assault", "Theft", "Vandalism", "Trespassing"]
    
    if top_class in security_threats and top_confidence > 70:
        threat_detected = True
        
        # Determine severity based on confidence level and type
        if top_class in ["Robbery", "Assault"] and top_confidence > 90:
            severity = "Critical"
        elif top_confidence > 85:
            severity = "High"
        elif top_confidence > 75:
            severity = "Medium"
            
        details = f"Detected {top_class} with {top_confidence:.2f}% confidence. {gemini_analysis}"
    
    return {
        "threat_detected": threat_detected,
        "severity": severity,
        "details": details,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route("/process_security_analysis", methods=["POST"])
def process_security_analysis():
    try:
        # Get data from request
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request, no JSON data received"}), 400
        
        gemini_analysis = data.get("gemini_analysis")
        model_analysis = data.get("model_analysis")
        notification_settings = data.get("notification_settings", {})
        
        # Extract notification settings or use defaults
        whatsapp_recipient = notification_settings.get("whatsapp_number", "")
        email_recipient = notification_settings.get("email", "")
        notification_threshold = notification_settings.get("threshold", "Medium")  # Low, Medium, High, Critical
        
        # Validate required data
        if not gemini_analysis or not model_analysis:
            return jsonify({"error": "Missing required analysis data"}), 400
            
        # Evaluate the security threat
        evaluation = evaluate_security_threat(model_analysis, gemini_analysis)
        
        # Determine if notifications should be sent based on severity threshold
        should_notify = False
        severity_levels = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        
        if evaluation["threat_detected"] and severity_levels.get(evaluation["severity"], 0) >= severity_levels.get(notification_threshold, 0):
            should_notify = True
        
        # Prepare response data
        response_data = {
            "evaluation": evaluation,
            "notifications": {"whatsapp": None, "email": None}
        }
        should_notify = True
        # Send notifications if needed
        if should_notify:
            print("Sending notifications...")
            # Prepare notification message
            notification_message = (
                f"SECURITY ALERT - {evaluation['severity']} Severity\n\n"
                f"Incident Type: {model_analysis['top_classes'][0]}\n"
                f"Confidence: {model_analysis['confidences'][0]:.2f}%\n"
                f"Time: {evaluation['timestamp']}\n\n"
                f"Details: {evaluation['details']}\n\n"
                f"Please check security system immediately."
            )
            print(whatsapp_recipient)
            # Send WhatsApp notification
            if whatsapp_recipient:
                print("Sending WhatsApp notification...")
                whatsapp_result = send_whatsapp_alert(whatsapp_recipient, notification_message)
                print("WhatsApp result:", whatsapp_result)
                response_data["notifications"]["whatsapp"] = whatsapp_result
            
            # Send email notification
            if email_recipient:
                email_subject = f"SECURITY ALERT: {evaluation['severity']} - {model_analysis['top_classes'][0]} Detected"
                email_result = send_email_alert(email_recipient, email_subject, notification_message)
                response_data["notifications"]["email"] = email_result
                
        return jsonify(response_data), 200, {'Content-Type': 'application/json; charset=utf-8'}
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500, {'Content-Type': 'application/json; charset=utf-8'}

@app.route("/process_sensor_alert", methods=["POST"])
def process_sensor_alert():
    try:
        # Get data from request
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request, no JSON data received"}), 400
        
        # Extract notification settings or use defaults
        notification_settings = data.get("notification_settings", {})
        whatsapp_recipient = notification_settings.get("whatsapp_number", "")
        email_recipient = notification_settings.get("email", "")
        notification_threshold = notification_settings.get("threshold", "Medium")  # Low, Medium, High, Critical
        
        # Validate required data structure
        if "major_class" not in data or "detailed_class" not in data or "insights" not in data:
            return jsonify({"error": "Missing required sensor data structure"}), 400
            
        # Evaluate the sensor alert
        evaluation = evaluate_sensor_alert(data)
        
        # Determine if notifications should be sent based on severity threshold
        should_notify = False
        severity_levels = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        
        if evaluation["alert_triggered"] and severity_levels.get(evaluation["severity"], 0) >= severity_levels.get(notification_threshold, 0):
            should_notify = True
        
        # Prepare response data
        response_data = {
            "evaluation": evaluation,
            "notifications": {"whatsapp": None, "email": None}
        }
        
        # Send notifications if needed
        if should_notify:
            print(f"Sending notifications for {evaluation['alert_type']} alert...")
            
            # Create markdown report
            markdown_report = generate_markdown_report(data, evaluation)
            
            # Save markdown report to a file
            report_filename = f"sensor_alert_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_filename, 'w', encoding='utf-8') as report_file:
                report_file.write(markdown_report)
            
            # Create WhatsApp message (shorter form)
            whatsapp_message = (
                f"SENSOR ALERT - {evaluation['severity']} Severity\n\n"
                f"Alert Type: {evaluation['alert_type']}\n"
                f"Location: {data.get('location', 'Unknown')}\n"
                f"Time: {evaluation['timestamp']}\n\n"
                f"Key Metrics:\n"
                f"• {evaluation['key_metrics'][0]}\n"
                f"• {evaluation['key_metrics'][1]}\n"
                f"• {evaluation['key_metrics'][2]}\n\n"
                f"Recommended Action: {evaluation['recommended_action']}"
            )
            
            # Send WhatsApp notification
            if whatsapp_recipient:
                print("Sending WhatsApp notification...")
                whatsapp_result = send_whatsapp_alert(whatsapp_recipient, whatsapp_message)
                response_data["notifications"]["whatsapp"] = whatsapp_result
            
            # Send email notification with markdown report attached
            if email_recipient:
                email_subject = f"SENSOR ALERT: {evaluation['severity']} - {evaluation['alert_type']}"
                email_result = send_email_alert_with_attachment(email_recipient, email_subject, report_filename)
                response_data["notifications"]["email"] = email_result
                
        return jsonify(response_data), 200, {'Content-Type': 'application/json; charset=utf-8'}
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500, {'Content-Type': 'application/json; charset=utf-8'}

def send_email_alert_with_attachment(recipient, subject, attachment_path):
    """Send email notification with an attachment using Composio through LangChain agent"""
    try:
        result = agent.run(
            f"Send an email to {recipient} with the subject '{subject}' and attach the file at {attachment_path}. "
            f"Convert the markdown file at {attachment_path} to a .docx file before sending."
        )
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

def convert_markdown_to_docx(markdown_path):
    """Convert a markdown file to a .docx file"""
    
    # Read markdown content
    with open(markdown_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    # Create a new Document
    doc = Document()

    # Add HTML content to the document
    doc.add_paragraph(html_content)

    # Save the document as .docx
    docx_path = markdown_path.replace('.md', '.docx')
    doc.save(docx_path)

    return docx_path

def evaluate_sensor_alert(sensor_data):
    """Evaluate sensor data to determine alert severity and details"""
    # Extract key data
    major_class = sensor_data.get("major_class")
    major_class_name = sensor_data.get("major_class_name")
    detailed_class = sensor_data.get("detailed_class")
    detailed_class_name = sensor_data.get("detailed_class_name")
    insights = sensor_data.get("insights", {})
    
    # Initialize response
    alert_triggered = False
    severity = "Low"
    alert_type = detailed_class_name
    timestamp = insights.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    key_metrics = []
    detailed_analysis = ""
    recommended_action = ""
    
    # Check if this is an abnormal event (non-zero major class)
    if major_class != 0:
        alert_triggered = True
        
        # Extract risk scores for severity assessment
        risk_scores = insights.get("risk_scores", {})
        composite_indices = insights.get("composite_indices", {})
        
        # Determine severity based on various factors
        event_severity = composite_indices.get("event_severity_index", 0)
        env_impact = composite_indices.get("environmental_impact", 0)
        structural_risk = composite_indices.get("structural_risk_index", 0)
        
        # Calculate overall severity
        if event_severity > 7 or any(risk.get("severity") == "Critical" for risk in risk_scores.values() if isinstance(risk, dict)):
            severity = "Critical"
        elif event_severity > 5 or env_impact > 70 or any(risk.get("severity") == "High" for risk in risk_scores.values() if isinstance(risk, dict)):
            severity = "High"
        elif event_severity > 3 or env_impact > 50 or any(risk.get("severity") == "Medium" for risk in risk_scores.values() if isinstance(risk, dict)):
            severity = "Medium"
    
    # Handle specific alert types
    if detailed_class_name == "Power Outage":
        # Extract relevant metrics for power outage
        temp = insights.get("raw_sensors", {}).get("temperature", {}).get("value", 0)
        humidity = insights.get("raw_sensors", {}).get("humidity", {}).get("value", 0)
        thermal_risk = insights.get("risk_scores", {}).get("thermal_risk", {}).get("value", 0)
        
        key_metrics = [
            f"Temperature: {temp}°C",
            f"Humidity: {humidity}%",
            f"Thermal Risk: {thermal_risk}"
        ]
        
        detailed_analysis = (
            f"The power outage event shows a temperature of {temp}°C with humidity at {humidity}%. "
            f"The thermal risk factor is {thermal_risk}, suggesting {'moderate' if thermal_risk > 5 else 'minimal'} "
            f"environmental stress on equipment. "
            f"The temperature-humidity stress value is {insights.get('insights', {}).get('additional_insights', {}).get('temp_humidity_stress', 0):.2f}."
        )
        
        recommended_action = (
            f"Check backup power systems and monitor equipment temperature. "
            f"{'Immediate intervention required to prevent equipment damage.' if severity in ['High', 'Critical'] else 'Regular monitoring advised.'}"
        )
    
    # Add more alert type handling for other detailed_class_name values
    elif detailed_class_name == "Intrusion":
        motion = insights.get("raw_sensors", {}).get("ugs_motion", {}).get("value", 0)
        lidar = insights.get("raw_sensors", {}).get("lidar_distance", {}).get("value", 0)
        intrusion_risk = insights.get("risk_scores", {}).get("intrusion_risk", {}).get("value", 0)
        
        key_metrics = [
            f"Motion: {motion}",
            f"LIDAR Distance: {lidar}m",
            f"Intrusion Risk: {intrusion_risk}"
        ]
        
        detailed_analysis = (
            f"Potential intrusion detected with motion value of {motion} and LIDAR proximity of {lidar}m. "
            f"The intrusion risk assessment is {intrusion_risk}, indicating a "
            f"{'high' if intrusion_risk > 7 else 'moderate' if intrusion_risk > 4 else 'low'} security threat."
        )
        
        recommended_action = (
            f"{'Immediately dispatch security personnel.' if severity in ['High', 'Critical'] else 'Review security camera footage and increase monitoring.'}"
        )
    
    # Default case for other alert types
    else:
        # Extract top 3 most concerning metrics
        raw_sensors = insights.get("raw_sensors", {})
        metrics = []
        
        for sensor_name, sensor_data in raw_sensors.items():
            if isinstance(sensor_data, dict) and sensor_data.get("severity") not in ["N/A", "Low"]:
                metrics.append((sensor_name, sensor_data.get("value", 0), sensor_data.get("severity", "Low")))
        
        # Sort by severity and take top 3 (or fewer if not enough)
        metrics.sort(key=lambda x: {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}.get(x[2], 0), reverse=True)
        top_metrics = metrics[:3]
        
        # If we don't have 3 concerning metrics, add some standard ones
        while len(top_metrics) < 3:
            for sensor_name, sensor_data in raw_sensors.items():
                if isinstance(sensor_data, dict) and (sensor_name, sensor_data.get("value", 0), sensor_data.get("severity", "Low")) not in top_metrics:
                    top_metrics.append((sensor_name, sensor_data.get("value", 0), sensor_data.get("severity", "Low")))
                    break
        
        # Format key metrics
        key_metrics = [f"{name.replace('_', ' ').title()}: {value}" for name, value, _ in top_metrics[:3]]
        
        # Generic analysis and recommendation
        detailed_analysis = (
            f"Alert triggered for {detailed_class_name} with severity {severity}. "
            f"Event severity index is {composite_indices.get('event_severity_index', 0):.2f} and "
            f"environmental impact is rated at {composite_indices.get('environmental_impact', 0):.2f}."
        )
        
        recommended_action = (
            f"{'Immediate investigation required.' if severity in ['High', 'Critical'] else 'Monitor system and check for anomalies.'}"
        )
    
    return {
        "alert_triggered": alert_triggered,
        "severity": severity,
        "alert_type": alert_type,
        "timestamp": timestamp,
        "key_metrics": key_metrics,
        "detailed_analysis": detailed_analysis,
        "recommended_action": recommended_action
    }

def generate_markdown_report(sensor_data, evaluation):
    """Generate a detailed markdown report for the sensor alert"""
    insights = sensor_data.get("insights", {})
    raw_sensors = insights.get("raw_sensors", {})
    risk_scores = insights.get("risk_scores", {})
    composite_indices = insights.get("composite_indices", {})
    
    # Create markdown report
    markdown = f"""
# Sensor Alert Report: {evaluation['alert_type']}

## Alert Summary
- **Severity**: {evaluation['severity']}
- **Alert Type**: {evaluation['alert_type']}
- **Timestamp**: {evaluation['timestamp']}
- **Status**: {'🚨 ACTIVE' if evaluation['alert_triggered'] else '✓ RESOLVED'}

## Key Findings
{evaluation['detailed_analysis']}

## Sensor Readings

| Sensor | Value | Status |
|--------|-------|--------|
"""

    # Add sensor readings
    for sensor_name, sensor_data in raw_sensors.items():
        if isinstance(sensor_data, dict):
            value = sensor_data.get("value", "N/A")
            severity = sensor_data.get("severity", "N/A")
            color = sensor_data.get("color", "white")
            
            # Format name for display
            display_name = sensor_name.replace("_", " ").title()
            
            markdown += f"| {display_name} | {value} | {severity} |\n"
    
    # Add risk assessment
    markdown += f"""
## Risk Assessment

| Risk Factor | Value | Severity |
|-------------|-------|----------|
"""

    for risk_name, risk_data in risk_scores.items():
        if isinstance(risk_data, dict):
            value = risk_data.get("value", "N/A")
            severity = risk_data.get("severity", "N/A")
            
            # Format name for display
            display_name = risk_name.replace("_", " ").title()
            
            markdown += f"| {display_name} | {value} | {severity} |\n"
    
    # Add composite indices
    markdown += f"""
## Composite Indices

| Index | Value |
|-------|-------|
"""

    for index_name, value in composite_indices.items():
        # Format name for display
        display_name = index_name.replace("_", " ").title()
        
        # Format value
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        
        markdown += f"| {display_name} | {formatted_value} |\n"
    
    # Add recommended actions
    markdown += f"""
## Recommended Actions
{evaluation['recommended_action']}

---
*This report was automatically generated by the Sensor Alert System at {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""

    return markdown

if __name__ == "__main__":
    app.run(debug=True, port=3000)