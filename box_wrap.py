#helper function to create box around the text 
import textwrap
def create_boxed_text(text, box_char='#'):
    lines = text.split('\n')
    max_length = max(len(line) for line in lines)
    top_bottom = box_char * (max_length + 4)
    return "\n".join([top_bottom] + [f"{box_char} {line.ljust(max_length)} {box_char}" for line in lines] + [top_bottom])