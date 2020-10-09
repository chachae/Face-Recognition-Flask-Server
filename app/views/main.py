from flask import Blueprint, render_template

# 创建视图对象
main = Blueprint('main', __name__, template_folder='templates')


@main.route('/', methods=['GET'])
def main_page():
    return render_template(template_name_or_list='compare.html')
