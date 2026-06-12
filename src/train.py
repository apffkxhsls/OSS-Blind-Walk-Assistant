from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    print("🚀 YOLOv8 Custom 데이터셋 학습을 시작합니다...")
    
    # 이 부분은 나중에 data.yaml이 생성되면 주석을 풀고 실행
    # model.train(data='../data/custom_dataset/data.yaml', epochs=50, imgsz=640)

if __name__ == "__main__":
    main()