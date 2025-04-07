# IoT Smart Door Lock System

A intelligent door security system using RaspberryPi, camera recognition, and ThingsBoard for remote monitoring and control.

## Overview

This IoT-based smart door lock system provides automated visitor recognition with remote monitoring capabilities. The system captures images when visitors approach, processes them for face recognition, and either grants automatic access to authorized visitors or alerts homeowners about unauthorized visitors.

## Features

- Motion-triggered visitor detection
- Facial recognition for authorized users
- Real-time notifications to homeowners
- Remote door control through mobile app
- Image capture and storage of unauthorized visitors
- Access logs with timestamps
- ThingsBoard dashboard for system monitoring

## System Architecture

The system consists of:
- Camera module for visitor detection
- RaspberryPi for image processing and control
- ThingsBoard platform for data management and user interface
- Servo motor for physical lock control
- MQTT protocol for communication

## Installation

### Hardware Requirements
- Raspberry Pi 4 (2GB+ RAM)
- Pi Camera Module
- Servo Motor
- Door Lock Mechanism
- Motion Sensor (optional)
- Power Supply

### Software Setup
1. Clone this repository:
   ```
   git clone https://github.com/username/iot-smart-door-lock.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure ThingsBoard:
   - Create a new device in ThingsBoard
   - Configure rule chains for visitor notification
   - Set up dashboard for monitoring

4. Update configuration:
   ```
   cp config.example.json config.json
   # Edit config.json with your settings
   ```

5. Run the setup script:
   ```
   python setup.py
   ```

## Usage

### Starting the System
```
python main.py
```

### Training Face Recognition
```
python train_faces.py --name "Person Name" --images /path/to/images
```

### ThingsBoard Dashboard
Access the dashboard at `http://your-thingsboard-url/dashboard/YOUR_DASHBOARD_ID`

## Rule Chains

The system uses the following ThingsBoard rule chains:
- Root Rule Chain: For general message routing and processing
- Visitor Notification Rule Chain: For processing visitor detections and sending alerts

## Configuration Options

Edit `config.json` to customize:
- Face recognition confidence threshold
- Notification preferences
- Camera settings
- Access control rules

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- IoT platform by [ThingsBoard](https://thingsboard.io/)
