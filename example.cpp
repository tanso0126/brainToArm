#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;
Servo servo7;

void setup() {
  servo1.attach(13);
  servo2.attach(12);
  servo3.attach(11);//사용하지 않는 모터
  servo4.attach(10);
  servo5.attach(9);
  servo6.attach(8);
  servo7.attach(7);
}

void loop() { // roop -> loop 로 수정
  servo1.write(90);
  servo2.write(90);
  servo4.write(90);
  servo5.write(90);
  servo6.write(170);
  servo7.write(180);
  delay(100);    // 1초 대기
  
  servo1.write(100);
  servo7.write(90);
  delay(100);    // 동작을 확실히 하기 위해 1초 대기 추가
}