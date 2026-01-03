#include <math.h>

// ------------------ Motor Pins ------------------
// Right Front
const int RP_R_FRONT = 2;
const int LP_R_FRONT = 3;

// Left Front
const int RP_L_FRONT = 5;
const int LP_L_FRONT = 6;

// Left Back
const int RP_L_BACK = 9;
const int LP_L_BACK = 10;

// Right Back
const int RP_R_BACK = 11;
const int LP_R_BACK = 12;
// -------------------------------------------------

void setup() {
  Serial.begin(115200);

  pinMode(RP_R_FRONT, OUTPUT);
  pinMode(LP_R_FRONT, OUTPUT);
  pinMode(RP_L_FRONT, OUTPUT);
  pinMode(LP_L_FRONT, OUTPUT);
  pinMode(RP_L_BACK, OUTPUT);
  pinMode(LP_L_BACK, OUTPUT);
  pinMode(RP_R_BACK, OUTPUT);
  pinMode(LP_R_BACK, OUTPUT);

  stopAllMotors();
  Serial.println("X-Drive Ready");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "S") {
      stopAllMotors();
      Serial.println("STOP");
      return;
    }

    int commaIndex = cmd.indexOf(',');
    if (commaIndex > 0) {
      int angle = cmd.substring(0, commaIndex).toInt();
      int power = cmd.substring(commaIndex + 1).toInt();

      driveAngle(angle, power);
    }
  }
}

// ------------------ Motor Control ------------------
void driveAngle(float angle, int power) {
  // Add 180 degrees to fix forward/backward direction
  float rad = (angle + 180) * PI / 180.0;

  // Calculate speeds for X-drive
  float rf = -sin(rad) + cos(rad);
  float lf = -sin(rad) - cos(rad);
  float lb =  sin(rad) - cos(rad);
  float rb =  sin(rad) + cos(rad);

  // Normalize speeds
  float maxVal = max(max(abs(rf), abs(lf)), max(abs(lb), abs(rb)));
  if (maxVal > 1.0) {
    rf /= maxVal;
    lf /= maxVal;
    lb /= maxVal;
    rb /= maxVal;
  }

  int speed_rf = (int)(rf * power);
  int speed_lf = (int)(lf * power);
  int speed_lb = (int)(lb * power);
  int speed_rb = (int)(rb * power);

  Serial.print("RF: "); Serial.print(speed_rf);
  Serial.print(" LF: "); Serial.print(speed_lf);
  Serial.print(" LB: "); Serial.print(speed_lb);
  Serial.print(" RB: "); Serial.println(speed_rb);

  setMotor(RP_R_FRONT, LP_R_FRONT, speed_rf);
  setMotor(RP_L_FRONT, LP_L_FRONT, speed_lf);
  setMotor(RP_L_BACK, LP_L_BACK, speed_lb);
  setMotor(RP_R_BACK, LP_R_BACK, speed_rb);
}

void setMotor(int rpwmPin, int lpwmPin, int speed) {
  if (speed > 0) {
    analogWrite(rpwmPin, speed);
    analogWrite(lpwmPin, 0);
  } else if (speed < 0) {
    analogWrite(rpwmPin, 0);
    analogWrite(lpwmPin, -speed);
  } else {
    analogWrite(rpwmPin, 0);
    analogWrite(lpwmPin, 0);
  }
}

void stopAllMotors() {
  setMotor(RP_R_FRONT, LP_R_FRONT, 0);
  setMotor(RP_L_FRONT, LP_L_FRONT, 0);
  setMotor(RP_L_BACK, LP_L_BACK, 0);
  setMotor(RP_R_BACK, LP_R_BACK, 0);
}
