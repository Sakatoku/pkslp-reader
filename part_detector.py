import os
import sys
import cv2

# テストデータのファイル名
TEST_FILE = "test/Screenshot_20230810-014122.png"

# 画像をデバッグ表示する関数
def debug_show_image(img):
    # 画像を50%サイズで表示する
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow("image", img)
    # キー入力を待つ
    cv2.waitKey(0)
    # 表示したウィンドウを閉じる
    cv2.destroyAllWindows()

# 画像の読み込み
def read_image(filename):
    # OpenCVを使って画像を読み込む
    img = None
    img = cv2.imread(filename)
    return img

# 二値化
def binarize(img):
    # グレースケールに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二値化
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 二値化した画像を返す
    return img

# 輪郭が何階層目かを計算する
def get_hierarchy_levels(hierarchy):
    # hierarchyをflattenする
    hierarchy = hierarchy[0]
    # バッファを初期化
    hierarchy_levels = []
    for i in range(len(hierarchy)):
        hierarchy_levels.append(-1)
    # 0番目から順番に階層を計算する。計算した階層はバッファに格納する
    # 最も上の階層は0とする
    for i in range(len(hierarchy)):
        # 階層の深さを計算する
        level = 0
        j = i
        while hierarchy[j][3] != -1:
            level += 1
            j = hierarchy[j][3]
        # 計算した階層をバッファに格納する
        hierarchy_levels[i] = level
    return hierarchy_levels

# 輪郭抽出
def extract_contours(img):
    # 輪郭を抽出
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭が何階層目かを計算する
    hierarchy_levels = get_hierarchy_levels(hierarchy)

    # hierarchy_levelsを使って、第2階層目にある輪郭を抽出する
    contours = [contours[i] for i in range(len(contours)) if hierarchy_levels[i] == 1]

    # 面積が上位3つの輪郭を抽出する
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # 輪郭を返す
    return contours

# 日付領域とグラフ領域の輪郭の判定
def judge_contours(contours):
    # 輪郭の上部のY軸位置を取得する
    top_positions = []
    for i in range(len(contours)):
        # 輪郭ごとに最も上部のY軸位置を取得する
        y = 999999
        for j in range(len(contours[i])):
            if contours[i][j][0][1] < y:
                y = contours[i][j][0][1]
        # Y軸位置をバッファに格納する
        top_positions.append(y)
    
    # 面積を計算する
    areas = [cv2.contourArea(contours[i]) for i in range(len(contours))]

    # Y軸位置が最も小さい輪郭を日付領域とする
    date_index = top_positions.index(min(top_positions))
    date_part = contours[date_index]
    date_area = areas[date_index]

    # 日付領域よりも面積が大きく、Y軸位置が最も大きい輪郭をグラフ領域とする
    graph_index = -1
    graph_part = None
    for i in range(len(contours)):
        # 日付領域は除外する
        if i == date_index:
            continue
        # 面積が日付領域よりも大きいかを判定する
        if areas[i] > date_area:
            # 最初の要素ならそのまま格納する
            if graph_index == -1:
                graph_index = i
                graph_part = contours[i]
                continue
            # 既に探索した要素よりもY軸位置が大きいかを判定して、大きければ格納する
            if top_positions[i] > top_positions[graph_index]:
                graph_index = i
                graph_part = contours[i]

    return date_part, graph_part

# 輪郭を含む矩形の取得
def get_bounding_rect(contour):
    # 輪郭を含む矩形を計算する
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

# 与えられた画像から日付領域とグラフ領域を検出する関数
def detect_parts(img, debug=False):
    # 二値化
    img_bin = binarize(img)
    if debug:
        debug_show_image(img_bin)

    # 輪郭抽出
    contours = extract_contours(img_bin)
    if debug:
        # 輪郭を描画した画像を作成
        imt_tmp = img.copy()
        img_contours = cv2.drawContours(imt_tmp, contours, -1, (0, 255, 0), 3)
        debug_show_image(img_contours)

    # 日付領域とグラフ領域の輪郭を判定する
    date_part, graph_part = judge_contours(contours)
    if debug:
        # 輪郭を描画した画像を作成
        imt_tmp = img.copy()
        img_contours = cv2.drawContours(imt_tmp, [date_part], -1, (0, 255, 0), 3)
        img_contours = cv2.drawContours(imt_tmp, [graph_part], -1, (0, 0, 255), 3)
        debug_show_image(img_contours)

    # 各領域を含む矩形を取得する
    date_x, date_y, date_w, date_h = get_bounding_rect(date_part)
    graph_x, graph_y, graph_w, graph_h = get_bounding_rect(graph_part)
    if debug:
        # 各領域を含む矩形を塗りつぶした画像を作成
        img_rect = img.copy()
        img_rect = cv2.rectangle(img_rect, (date_x, date_y), (date_x + date_w, date_y + date_h), (0, 255, 0), -1)
        img_rect = cv2.rectangle(img_rect, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 255), -1)
        # 元の画像と0.5の透明度で重ねる
        img_rect = cv2.addWeighted(img, 0.5, img_rect, 0.5, 0)
        debug_show_image(img_rect)

    # 各領域を切り出す
    date_img = img[date_y:date_y + date_h, date_x:date_x + date_w]
    graph_img = img[graph_y:graph_y + graph_h, graph_x:graph_x + graph_w]
    if debug:
        debug_show_image(date_img)
        debug_show_image(graph_img)
    
    return date_img, graph_img

# 画像を保存する関数
def save_image(img, filename):
    cv2.imwrite(filename, img)

# メイン関数
def main(debug=False):
    # 引数からファイル名を取得
    filename = ""
    args = sys.argv
    if len(args) < 2:
        print("Error: No filename")
        if not debug:
            return
        else:
            filename = TEST_FILE
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
    if debug:
        debug_show_image(img)
    date_img, graph_img = detect_parts(img, debug=debug)

    # 日付領域とグラフ領域を保存する
    save_image(date_img, "output/date.png")
    save_image(graph_img, "output/graph.png")

if __name__ == "__main__":
    main(debug=True)
