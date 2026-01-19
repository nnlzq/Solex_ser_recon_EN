
pyinstaller -F SHG_MAIN.py --add-data=language_data:language_data --add-data=line_data:line_data


rem  nuitka --onefile SHG_MAIN.py  --standalone --msvc=latest  --enable-plugin=tk-inter    --include-data-dir=language_data=language_data --include-data-dir=line_data=line_data






