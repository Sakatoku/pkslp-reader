import os
import sys

# 画像の読み込み
def read_image(filename):
    print("read_image")
    return None

# コントラストの調整

# 二値化

# 輪郭抽出

# 面積が最大の輪郭を抽出

# 与えられた画像から日付領域とグラフ領域を検出する関数
def detect_parts(img):
    print("detect_parts")

# メイン関数
def main():
    # 引数からファイル名を取得
    filename = ""
    args = sys.argv
    if len(args) < 2:
        print("Error: No filename")
        return
    else:
        filename = args[1]

    # ファイル名が指定されていない場合はエラーを出力して終了
    if filename == "":
        print("Error: No filename")
        return

    # ファイルが存在しない場合はエラーを出力して終了
    if not os.path.exists(filename):
        print("Error: File not found")
        return

    # 画像から日付領域とグラフ領域を検出する
    img = read_image(filename)
    detect_parts(img)

if __name__ == "__main__":
    main()
