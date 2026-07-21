#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;

void setup() {
  servo1.attach(13);
  servo2.attach(12);
  servo3.attach(11);
  servo4.attach(10);
  servo5.attach(9);
  servo6.attach(8);
}

void loop() { // roop -> loop 로 수정
  servo1.write(90);
  servo2.write(70);
  servo3.write(90);
  servo4.write(90);
  servo5.write(0);
  servo6.write(180);
  delay(100);    // 0.1초 대기
  
  servo1.write(100);
  servo6.write(90);
  delay(100);    // 동작 확인을 위한 0.1초 대기
}
