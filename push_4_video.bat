@echo off
title FFMPEG Push 4 Streams via TCP
start "Cam 1" ffmpeg -re -stream_loop -1 -i "D:\ProjectAtin\cam_ai\assets\videos\camera_1.mp4" -map 0:v:0 -c copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/cam1
timeout /t 3 /nobreak >nul
start "Cam 2" ffmpeg -re -stream_loop -1 -i "D:\ProjectAtin\cam_ai\assets\videos\camera_1.mp4" -map 0:v:0 -c copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/cam2
timeout /t 3 /nobreak >nul
start "Cam 3" ffmpeg -re -stream_loop -1 -i "D:\ProjectAtin\cam_ai\assets\videos\camera_1.mp4" -map 0:v:0 -c copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/cam3
timeout /t 3 /nobreak >nul
start "Cam 4" ffmpeg -re -stream_loop -1 -i "D:\ProjectAtin\cam_ai\assets\videos\camera_1.mp4" -map 0:v:0 -c copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/cam4
echo Dang day 4 luong video qua giao thuc TCP...
pause