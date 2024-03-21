import cv2
import os

def main():
    videos13 = ['1-13/Videos/' + x for x in os.listdir('1-13/Videos')]
    videos39 = ['14-39/Videos/' + x for x in os.listdir('14-39/Videos')]
    videos52 = ['40-52/Videos/' + x for x in os.listdir('40-52/Videos')]
    
    video_to_image(videos13, 'images/1-13')
    video_to_image(videos39, 'images/14-39')
    video_to_image(videos52, 'images/40-52')

def video_to_image(path_list, save_destination=''):
    """
    Takes a list of paths to videos and splits each video in turn into its constituent images
    """
    for video in path_list:
        videocv = cv2.VideoCapture(video)
        count = 0 

        while True:
            success, image = videocv.read() # tracks succes of reading 
            if not success: # if there isn't a next frame we break
                break
            image_filename = f"{save_destination}/{video.split('/')[2].split('.')[0]}_frame_{count:04}.jpg.jpg"
            cv2.imwrite(image_filename, image)
            count += 1 # iterating to next frame 
            
        videocv.release()
        
if __name__ == "__main__":
    main()