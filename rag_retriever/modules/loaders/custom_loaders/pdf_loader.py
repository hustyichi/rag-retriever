import re
from typing import List

import cv2
import fitz
import numpy as np
import tqdm
from langchain_community.document_loaders import UnstructuredFileLoader
from PIL import Image

from rag_retriever.modules.loaders.custom_loaders.ocr import get_ocr

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR
PDF_OCR_THRESHOLD = (0.6, 0.6)


def is_chinese(text):
    chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
    if chinese_char_pattern.search(text):
        return True
    else:
        return False


def check_error_pdf(doc):
    for idx, page in enumerate(doc):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8

        if is_chinese(text) and "，" not in text and "。" not in text:
            return False

        if idx >= 5:
            break

    return True


# 根据实际发现的编码问题进行转换
def fix_error_pdf_content(text: str):
    # 匹配，和袁的映射
    text = text.replace("袁", "，")
    # 匹配。和 遥
    text = text.replace("遥", "。")
    # 匹配：和 院
    ## 如果院前面是医的话，那么就不用转换
    text = re.sub(r"(?<!医)院", "：", text)
    # 匹配（和 渊
    text = text.replace("渊", "（")
    # 匹配）和 冤
    text = text.replace("冤", "）")
    # 匹配、和尧
    text = text.replace("尧", "、")
    # 匹配【和 揖
    text = text.replace("揖", "【")
    # 匹配】和 铱
    text = text.replace("铱", "】")
    # 匹配℃ 和益 利用正则匹配了益字前面是否为数字，如果是数字那么才匹配
    # 注意识别出来的益和前面的数字之间有一个空格的
    text = re.sub(r"(?<=\d\s)益", "℃", text)
    # 匹配~和 耀
    text = text.replace("耀", "~")
    # 匹配；和 曰
    text = text.replace("曰", "；")

    text = re.sub(r"(\d)依", r"\1±", text)

    text = text.replace("滋g", "μg")

    text = re.sub(r"伊(\d+)", r"x\1", text)

    text = text.replace("覬", "∅")

    # 修复 《 和 》 解析异常
    text = re.sub(r"叶(.*?)曳", r"《\1》", text, flags=re.DOTALL)

    # 修复 ≤
    text = text.replace("臆", "≤")

    # 修复 -
    text = text.replace("鄄", "-")

    # 修复 ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ ⑩
    text = re.sub(
        r"淤(.*?)于(.*?)盂(.*?)榆(.*?)虞(.*?)愚(.*?)舆(.*?)余(.*?)俞(.*?)逾",
        r"①\1②\2③\3④\4⑤\5⑥\6⑦\7⑧\8⑨\9⑩",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"淤(.*?)于(.*?)盂(.*?)榆(.*?)虞(.*?)愚(.*?)舆(.*?)余(.*?)俞",
        r"①\1②\2③\3④\4⑤\5⑥\6⑦\7⑧\8⑨",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"淤(.*?)于(.*?)盂(.*?)榆(.*?)虞(.*?)愚(.*?)舆(.*?)余",
        r"①\1②\2③\3④\4⑤\5⑥\6⑦\7⑧",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"淤(.*?)于(.*?)盂(.*?)榆(.*?)虞(.*?)愚(.*?)舆",
        r"①\1②\2③\3④\4⑤\5⑥\6⑦",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"淤(.*?)于(.*?)盂(.*?)榆(.*?)虞(.*?)愚",
        r"①\1②\2③\3④\4⑤\5⑥",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"淤(.*?)于(.*?)盂(.*?)榆(.*?)虞", r"①\1②\2③\3④\4⑤", text, flags=re.DOTALL
    )
    text = re.sub(r"淤(.*?)于(.*?)盂(.*?)榆", r"①\1②\2③\3④", text, flags=re.DOTALL)
    text = re.sub(r"淤(.*?)于(.*?)盂", r"①\1②\2③", text, flags=re.DOTALL)
    text = re.sub(r"淤(.*?)于", r"①\1②", text, flags=re.DOTALL)

    # 修复 [ 和 ] 解析异常
    text = re.sub(r"咱(.{0,30}?)暂", r"[\1]", text, flags=re.DOTALL)

    # 修复罗马数字字符
    text = re.sub(r"(?<![\u4e00-\u9fa5])玉|玉(?![\u4e00-\u9fa5])|玉(?=期)", "Ⅰ", text)
    text = re.sub(r"(?<![\u4e00-\u9fa5])域|域(?![\u4e00-\u9fa5])|域(?=期)", "Ⅱ", text)
    text = re.sub(r"(?<![\u4e00-\u9fa5])芋|芋(?![\u4e00-\u9fa5])|芋(?=期)", "Ⅲ", text)
    text = re.sub(r"(?<![\u4e00-\u9fa5])郁|郁(?![\u4e00-\u9fa5])|郁(?=期)", "Ⅳ", text)

    return text


class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            """
            img   --image
            angle --rotation angle
            return--rotated img
            """

            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            # 获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img

        def pdf2text(filepath, do_ocr: bool = True):
            ocr = None
            if do_ocr:
                ocr = get_ocr()

            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(
                total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0"
            )
            for i, page in enumerate(doc):
                b_unit.set_description(
                    "RapidOCRPDFLoader context page index: {}".format(i)
                )
                b_unit.refresh()
                text = page.get_text("")

                resp += text + "\n"

                if not ocr:
                    continue

                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        # 检查图片尺寸是否超过设定的阈值
                        if (bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[
                            0
                        ] or (bbox[3] - bbox[1]) / (
                            page.rect.height
                        ) < PDF_OCR_THRESHOLD[
                            1
                        ]:
                            continue
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples
                        if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array)
                            ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360 - page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)

                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            resp = fix_error_pdf_content(resp)
            return resp

        text = pdf2text(self.file_path, do_ocr=False)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)
