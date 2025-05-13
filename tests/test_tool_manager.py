import unittest
import os
import shutil
import re
from src.tool_manager import (
    ListDirectoryTool,
    ReadFileContentTool,
    ToolPathNotFoundError,
    ToolPathIsNotDirectoryError,
    ToolPathIsNotFileError,
    ToolInvalidArgumentError,
    ToolExecutionError # Assuming this might be raised for very generic OS errors if not caught by others
)

class TestToolManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，创建测试所需的文件和目录结构"""
        # 使用一个更独特的顶级测试目录名，以减少与其他测试冲突的可能性
        cls.base_test_dir = "tests/temp_test_data_for_tool_manager"
        cls.empty_dir = os.path.join(cls.base_test_dir, "test_empty_dir")
        cls.dir_with_files = os.path.join(cls.base_test_dir, "test_dir_with_files")
        cls.file1_path = os.path.join(cls.dir_with_files, "file1.txt")
        cls.file2_path = os.path.join(cls.dir_with_files, "file2.py")
        cls.single_file_path = os.path.join(cls.base_test_dir, "test_single_file.txt")
        cls.unreadable_file_path = os.path.join(cls.base_test_dir, "unreadable.txt") # For permission errors if possible to simulate

        # 清理可能存在的旧测试目录
        if os.path.exists(cls.base_test_dir):
            shutil.rmtree(cls.base_test_dir)

        # 创建目录
        os.makedirs(cls.empty_dir, exist_ok=True)
        os.makedirs(cls.dir_with_files, exist_ok=True)

        # 创建文件
        with open(cls.file1_path, 'w', encoding='utf-8') as f:
            f.write("Hello World")
        with open(cls.file2_path, 'w', encoding='utf-8') as f:
            f.write("print('Python')")
        with open(cls.single_file_path, 'w', encoding='utf-8') as f:
            f.write("This is a test file with multiple lines.\nLine 2.\nLine 3 here.")
        
        # 尝试创建一个不可读文件 (在某些系统上可能不起作用或需要特定权限)
        # try:
        #     with open(cls.unreadable_file_path, 'w') as f:
        #         f.write("noperm")
        #     os.chmod(cls.unreadable_file_path, 0o000) # No read/write/execute
        # except OSError:
        #     cls.unreadable_file_path = None # Mark as unavailable if creation fails

    @classmethod
    def tearDownClass(cls):
        """在所有测试结束后，清理创建的测试文件和目录"""
        # 还原权限以便删除 (如果unreadable_file_path测试被激活且成功设置权限)
        # if cls.unreadable_file_path and os.path.exists(cls.unreadable_file_path):
        #     try:
        #        os.chmod(cls.unreadable_file_path, 0o600)
        #     except OSError:
        #         pass # Ignore if chmod fails
        if os.path.exists(cls.base_test_dir):
            shutil.rmtree(cls.base_test_dir)

    def setUp(self):
        """每个测试方法执行前调用"""
        self.list_dir_tool = ListDirectoryTool()
        self.read_file_tool = ReadFileContentTool()

    # --- ListDirectoryTool Tests ---
    def test_list_directory_success(self):
        result = self.list_dir_tool.run(self.dir_with_files)
        self.assertIsInstance(result, list)
        self.assertCountEqual(result, ["file1.txt", "file2.py"]) # Order doesn't matter

    def test_list_empty_directory(self):
        result = self.list_dir_tool.run(self.empty_dir)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_list_directory_path_not_exists(self):
        with self.assertRaises(ToolPathNotFoundError):
            self.list_dir_tool.run("non_existent_dir_path_for_test")

    def test_list_directory_path_is_file(self):
        with self.assertRaises(ToolPathIsNotDirectoryError):
            self.list_dir_tool.run(self.single_file_path)
    
    # --- ReadFileContentTool Tests ---
    def test_read_file_content_all_lines_success(self):
        expected_content = "This is a test file with multiple lines.\nLine 2.\nLine 3 here."
        result = self.read_file_tool.run(self.single_file_path)
        self.assertEqual(result, expected_content)

    def test_read_file_content_specific_lines_success(self):
        expected_content_line2 = "Line 2.\n"
        result = self.read_file_tool.run(self.single_file_path, start_line=2, end_line=2)
        self.assertEqual(result, expected_content_line2)
        
        expected_content_lines1_2 = "This is a test file with multiple lines.\nLine 2.\n"
        result = self.read_file_tool.run(self.single_file_path, start_line=1, end_line=2)
        self.assertEqual(result, expected_content_lines1_2)
        
        expected_content_line2_onwards = "Line 2.\nLine 3 here."
        result = self.read_file_tool.run(self.single_file_path, start_line=2)
        self.assertEqual(result, expected_content_line2_onwards)

        expected_content_up_to_line1 = "This is a test file with multiple lines.\n"
        result = self.read_file_tool.run(self.single_file_path, end_line=1)
        self.assertEqual(result, expected_content_up_to_line1)

    def test_read_file_content_file_not_exists(self):
        with self.assertRaises(ToolPathNotFoundError):
            self.read_file_tool.run("non_existent_file_for_test.txt")

    def test_read_file_content_path_is_directory(self):
        with self.assertRaises(ToolPathIsNotFileError):
            self.read_file_tool.run(self.dir_with_files)

    def test_read_file_content_invalid_line_args_type(self):
        with self.assertRaises(ToolInvalidArgumentError):
            self.read_file_tool.run(self.single_file_path, start_line="abc")
        with self.assertRaises(ToolInvalidArgumentError):
            self.read_file_tool.run(self.single_file_path, end_line="xyz")

    def test_read_file_content_invalid_line_range(self):
        with self.assertRaises(ToolInvalidArgumentError):
            self.read_file_tool.run(self.single_file_path, start_line=3, end_line=1)
        with self.assertRaises(ToolInvalidArgumentError):
            self.read_file_tool.run(self.single_file_path, end_line=-1)

    def test_read_file_content_start_line_out_of_bounds_returns_empty(self):
        # If start_line is specified and is beyond the number of lines, should return empty string
        result = self.read_file_tool.run(self.single_file_path, start_line=10)
        self.assertEqual(result, "")

    def test_read_file_content_end_line_out_of_bounds_caps_at_end(self):
        # end_line greater than total lines should effectively mean read to end of file
        expected_all_content = "This is a test file with multiple lines.\nLine 2.\nLine 3 here."
        result = self.read_file_tool.run(self.single_file_path, end_line=100)
        self.assertEqual(result, expected_all_content)

    def test_read_file_content_start_line_zero_or_negative_means_from_beginning(self):
        expected_first_line = "This is a test file with multiple lines.\n"
        result = self.read_file_tool.run(self.single_file_path, start_line=0, end_line=1)
        self.assertEqual(result, expected_first_line)
        result = self.read_file_tool.run(self.single_file_path, start_line=-5, end_line=1)
        self.assertEqual(result, expected_first_line)
    
    # def test_read_unreadable_file(self):
    #     if self.unreadable_file_path:
    #         with self.assertRaises(ToolExecutionError): # Or a more specific permission error if defined
    #             self.read_file_tool.run(self.unreadable_file_path)
    #     else:
    #         self.skipTest("无法创建不可读文件以进行测试 (可能由于权限或OS限制)。")

if __name__ == '__main__':
    unittest.main() 