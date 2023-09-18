from  moviepy.editor import *

def mp4_to_gif2(source):
    clip = VideoFileClip(source)
    clip.write_gif("./output2.gif")

def convert_gif_to_mp4(source):
    import moviepy.editor as mp
    clip = mp.VideoFileClip("demo.gif")
    clip.write_videofile("output.mp4")


from moviepy.editor import VideoFileClip


def convert_mp4_to_gif(mp4_file, gif_file):
    # 加载视频文件
    clip = VideoFileClip(mp4_file)

    # 将视频文件转换为 GIF 动画
    clip.write_gif(gif_file)






if __name__ == '__main__':

    # 示例用法
    mp4_file = "C:/Users/chenxinyi/Pictures/20230917213507.mp4"
    gif_file = "C:/Users/chenxinyi/Pictures/output.gif"
    convert_mp4_to_gif(mp4_file, gif_file)
    # mp4_to_gif2(source)

# clipVideo = VideoFileClip(r"C:/Users/chenxinyi/Pictures/20230917213507.mp4").crop(0, 278, 540, 580)
# clipVideo.write_gif(r"F:\video\WinBasedWorkHard.gif")