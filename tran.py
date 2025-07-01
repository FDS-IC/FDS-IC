import sys
import os


def convert_text_to_target_format(input_file, output_file, start_id=2):
    print(f"正在处理: 输入文件={input_file}, 输出文件={output_file}, 起始ID={start_id}")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在！")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            if not lines:
                print("警告：输入文件为空！")
                return

            with open(output_file, 'w', encoding='utf-8') as f_out:
                current_id = start_id
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    chars = [f'"{char}"' for char in line]
                    formatted_line = f'[*id": {current_id}, "text": [{", ".join(chars)}]]\n'
                    f_out.write(formatted_line)
                    current_id += 1

                print(f"成功写入 {len(lines)} 行到 {output_file}")

    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    input_file = 'D:/pythonProject/BERT-BILSTM-CRF-main/public _data/test.txt'
    output_file = 'D:/pythonProject/BERT-BILSTM-CRF-main/public _data/test2.txt'
    convert_text_to_target_format(input_file, output_file)