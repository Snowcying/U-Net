import matplotlib.pyplot as plt


def plot_line_chart(x, y,title):
    # 绘制折线图
    plt.plot(x, y)
    plt.plot([0] + x, [0] + y)
    plt.ylim(0, max(y))
    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('iou')

    # 显示图形
    plt.show()


def plot_2line_chart(x1, y1, x2, y2,title,y_label,x_label="epoch"):
    # 绘制第一条折线图
    plt.plot(x1, y1, label='train')

    # 绘制第二条折线图
    plt.plot(x2, y2, label='val')

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


def iouToDice(iou):
    dice = []
    for x in iou:
        dice.append((2 * x) / (x + 1))
    return dice


if __name__ == '__main__':

    # x
    train_data = [0, 0.374, 0.69, 0.772, 0.824, 0.858, 0.877, 0.894, 0.888, 0.913, 0.918, 0.917, 0.92, 0.928, 0.933, 0.916, 0.936, 0.937]
    val_data = [0,0.406, 0.608, 0.69, 0.68, 0.707, 0.706, 0.707, 0.706, 0.704, 0.71, 0.708, 0.707, 0.705, 0.708, 0.705, 0.706, 0.71]

    # y
    # train_data = [0, 0.374, 0.69, 0.772, 0.824, 0.858, 0.877, 0.894, 0.888, 0.913, 0.918, 0.917, 0.92, 0.928, 0.933, 0.916, 0.936, 0.937]
    # val_data = [0, 0.406, 0.608, 0.69, 0.68, 0.707, 0.706, 0.707, 0.706, 0.704, 0.71, 0.708, 0.707, 0.705, 0.708, 0.705, 0.706, 0.71]
    #
    # z
    # train_data = [0, 0.679, 0.749, 0.812, 0.845, 0.875, 0.884, 0.898, 0.902, 0.917, 0.922, 0.935, 0.936, 0.939, 0.926,
    #               0.942, 0.946]
    # val_data = [0, 0.659, 0.664, 0.665, 0.697, 0.701, 0.704, 0.71, 0.717, 0.719, 0.717, 0.718, 0.709, 0.725, 0.728,
    #             0.73, 0.731]

    train_dice_data = iouToDice(train_data)
    val_dice_data = iouToDice(val_data)
    print(val_dice_data)

    x = [i for i in range(0, len(val_data))]
    # plot_line_chart(x,train_data,"train_x_iou")
    # plot_line_chart(x,val_data,"val_x_iou")
    plot_2line_chart(x, train_data, x, val_data, "y_iou", "iou")
    plot_2line_chart(x, train_dice_data, x, val_dice_data, "y_dice", "dice")
