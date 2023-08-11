import os
import sys
import cv2

# テストデータのファイル名
TEST_FILE = "test/Screenshot_20230810-014122.png"

# 画像をデバッグ表示する関数
def debug_show_image(img):
    # タイムアウト時間(ミリ秒)を設定する。0の場合は無限に表示する
    timeout = 10000
    # 拡大率を設定する。1.0の場合はそのままのサイズで表示する
    scale = 0.3

    # 画像を表示する
    img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow("image", img)
    # キー入力を待つ
    # ウィンドウクローズしたときにキー入力できなくなりフリーズするため、timeout(ミリ秒)でタイムアウトさせる
    cv2.waitKey(timeout)

    # 表示したウィンドウを閉じる
    cv2.destroyAllWindows()

# 画像の読み込み
def read_image(filename):
    # OpenCVを使って画像を読み込む
    img = None
    img = cv2.imread(filename)
    return img

# 二値化
def binarize(img, theshold=-1):
    # グレースケールに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二値化する。thresholdが-1の場合は自動的に閾値を計算する
    if theshold < 0:
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    else:
        img = cv2.threshold(img, theshold, 255, cv2.THRESH_BINARY)[1]

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

    # hierarchy_levelsを使って、第2階層にある輪郭を抽出する
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

# 与えられた画像から日付領域とグラフ領域と横軸ラベル領域を抽出する関数
def split_parts(img, debug=False):
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

    # 日付領域とラベルを含むグラフ領域の輪郭を判定する
    date_part, graph_with_label_part = judge_contours(contours)
    if debug:
        # 輪郭を描画した画像を作成
        imt_tmp = img.copy()
        img_contours = cv2.drawContours(imt_tmp, [date_part], -1, (0, 255, 0), 3)
        img_contours = cv2.drawContours(imt_tmp, [graph_with_label_part], -1, (0, 0, 255), 3)
        debug_show_image(img_contours)

    # 各領域を含む矩形を取得する
    date_x, date_y, date_w, date_h = get_bounding_rect(date_part)
    graph_x, graph_y, graph_w, graph_h = get_bounding_rect(graph_with_label_part)
    if debug:
        # 各領域を含む矩形を塗りつぶした画像を作成
        img_rect = img.copy()
        img_rect = cv2.rectangle(img_rect, (date_x, date_y), (date_x + date_w, date_y + date_h), (0, 255, 0), -1)
        img_rect = cv2.rectangle(img_rect, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 255), -1)
        # 元の画像と0.5の透明度で重ねる
        img_rect = cv2.addWeighted(img, 0.5, img_rect, 0.5, 0)
        debug_show_image(img_rect)

    # 各領域を切り出す
    img_date = img[date_y:date_y + date_h, date_x:date_x + date_w]
    img_graph_with_label = img[graph_y:graph_y + graph_h, graph_x:graph_x + graph_w]
    if debug:
        debug_show_image(img_date)
        debug_show_image(img_graph_with_label)

    # ラベルを含むグラフ領域からグラフ領域と横軸ラベル領域を抽出する
    img_graph, img_label = split_graph_and_label(img_graph_with_label, debug=debug)

    return img_date, img_graph, img_label

# ラベルを含むグラフ領域からグラフ領域と横軸ラベル領域を抽出する関数
def split_graph_and_label(img_graph_with_label, debug=True):
    # 二値化
    img_bin = binarize(img_graph_with_label, 100)
    if debug:
        debug_show_image(img_bin)

    # 輪郭を抽出
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭が何階層目かを計算する
    hierarchy_levels = get_hierarchy_levels(hierarchy)

    # hierarchy_levelsを使って、第3階層にある輪郭を抽出する
    contours = [contours[i] for i in range(len(contours)) if hierarchy_levels[i] == 2]

    # 各輪郭のトップ位置とボトム位置を計算する
    top_positions, bottom_positions = get_top_bottom_positions(contours)
    if debug:
        print(top_positions)
        print(bottom_positions)

    # 一番面積が大きい輪郭がグラフ領域なので、面積が最上位の輪郭を抽出する
    # まず一番面積が大きい輪郭のインデックスを取得する
    line_contour_index = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)[:1][0]
    line_contour = contours[line_contour_index]
    if debug:
        # 輪郭を描画した画像を作成
        imt_tmp = img_graph_with_label.copy()
        img_contours = cv2.drawContours(imt_tmp, [line_contour], -1, (0, 255, 0), 3)
        debug_show_image(img_contours)
    # グラフ領域を含む矩形を計算する
    x, y, w, h = get_bounding_rect(line_contour)
    # グラフ領域のみを抽出する
    img_line = img_graph_with_label[y:y + h, x:x + w]
    if debug:
        debug_show_image(img_line)

    # 一番下にある輪郭が横軸ラベル領域なので、ボトム位置が最大の輪郭を確認する
    # まず一番下にある輪郭のインデックスを取得する
    label_contour_index = bottom_positions.index(max(bottom_positions))
    # グラフ領域のボトム位置と横軸ラベル領域のトップ位置の間の位置を計算する
    y1 = bottom_positions[line_contour_index]
    y2 = top_positions[label_contour_index]
    split_y = int((y1 + y2) / 2)
    # (0, split_y, width, height)の矩形で画像を分割して、下半分のみを抽出する
    width = img_graph_with_label.shape[1]
    height = img_graph_with_label.shape[0]
    img_label = img_graph_with_label[split_y:height, 0:width]
    if debug:
        debug_show_image(img_label)

    return img_line, img_label

# 画像を保存する関数
def save_image(img, filename):
    cv2.imwrite(filename, img)

# 各輪郭のトップ位置とボトム位置を計算する関数
def get_top_bottom_positions(contours):
    # 各輪郭のトップ位置とボトム位置を格納するリスト
    top_positions = []
    bottom_positions = []

    # リストを初期化する
    for _ in range(len(contours)):
        top_positions.append(999999)
        bottom_positions.append(-1)

    # 各輪郭のトップ位置とボトム位置を計算する
    for i, contour in enumerate(contours):
        # 輪郭を含む矩形を計算する
        _, y, _, h = get_bounding_rect(contour)
        # トップ位置とボトム位置を更新する
        if top_positions[i] > y:
            top_positions[i] = y
        if bottom_positions[i] < y + h:
            bottom_positions[i] = y + h

    return top_positions, bottom_positions

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

    # 画像から日付領域とグラフ領域と横軸ラベル領域を抽出する
    img = read_image(filename)
    if debug:
        debug_show_image(img)
    date_img, graph_img, label_img = split_parts(img, debug=debug)

    # 日付領域とグラフ領域と横軸ラベル領域を保存する
    save_image(date_img, "output/date.png")
    save_image(graph_img, "output/graph.png")
    save_image(label_img, "output/label.png")

if __name__ == "__main__":
    main(debug=True)
