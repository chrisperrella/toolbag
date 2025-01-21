import argparse
import asyncio
import multiprocessing
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Coroutine

class PythonFormatter:
    def __init__(self, args: List[str]):
        parser = argparse.ArgumentParser(
            description="Format all Python code (with black and isort).",
            prog="python_formatter",
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            default=False,
            help="Only list files and exit.",
        )
        parser.add_argument(
            "-v",
            "--verify",
            action="store_true",
            default=False,
            help="Verify formatting without applying changes.",
        )
        args = parser.parse_args(args=args)
        
        self.__list = args.list
        self.__verify = args.verify
        self.__failed_files = []
        self.__formatted_files = set()

    def run(self) -> int:
        files = self.__get_git_managed_python_files()
        if len(files) == 0:
            print("No Python files found to format.")
            return 0

        if self.__list:
            print(f"Found {len(files)} Python files:")
            print("\n".join(files))
            return 0

        print("Starting formatting tasks...")

        tasks_black = []
        tasks_isort = []
        for file in files:
            print(f"Adding isort task for: {file}")
            tasks_isort.append(self.__async_run_python_isort(file))
        
        for file_chunk in self.__chunk_files_for_black(files):
            print(f"Adding black task for chunk: {file_chunk}")
            tasks_black.append(self.__async_run_python_black(file_chunk))

        print("Executing tasks...")

        failed_files = []
        failed_files += asyncio.run(self.__run_tasks_in_parallel(multiprocessing.cpu_count(), tasks_isort))
        print("Completed isort tasks.")

        failed_files += asyncio.run(self.__run_tasks_in_parallel(1, tasks_black))
        print("Completed black tasks.")

        if failed_files:
            print(f"Formatting failed for {len(failed_files)} files.")
            for file in failed_files:
                print(file)
            return 1

        if self.__formatted_files:
            print(f"Successfully formatted {len(self.__formatted_files)} files.")

        return 0

    def __get_git_managed_python_files(self) -> List[str]:
        command = ["git", "ls-files", "--", "plugin/**/*.py"]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to retrieve git-managed files. Error: {result.stderr.strip()}")
            return []
        files = result.stdout.strip().split("\n")
        print(f"Git-managed files found: {files}")
        
        return [str(Path(f).resolve()) for f in files if f]

    def __chunk_files_for_black(self, files: List[str]) -> List[List[str]]:
        black_args = [
            sys.executable,
            "-m",
            "black",
            "-l",
            "88",  # Line length can be adjusted if needed
        ]
        if self.__verify:
            black_args.append("--check")

        chunk_size = 32000 - len(" ".join(black_args))
        chunks = []
        current_chunk = []
        current_size = 0

        for file in files:
            file_size = len(file) + 1
            if current_size + file_size > chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            current_chunk.append(file)
            current_size += file_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def __async_run_python_black(self, files: List[str]):
        args = [
            sys.executable,
            "-m",
            "black",
            "-l",
            "88",
            "--check" if self.__verify else "--quiet",
            *files,
        ]

        returncode, _ = await self.__run_subprocess(args)
        if returncode != 0:
            return files

        self.__formatted_files.update(files)
        return []

    async def __async_run_python_isort(self, file: str):
        args = [sys.executable, "-m", "isort", "--profile", "black", file]
        if self.__verify:
            args.append("--check")
        else:
            os.chmod(file, stat.S_IWRITE)

        returncode, _ = await self.__run_subprocess(args)
        if returncode != 0:
            return [file]

        self.__formatted_files.add(file)
        return []

    async def __run_subprocess(self, args: List[str]):
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode().strip()

    async def __run_tasks_in_parallel(self, job_count: int, tasks: List[Coroutine]) -> List[str]:
        semaphore = asyncio.Semaphore(job_count)

        async def run_task(task):
            async with semaphore:
                return await task

        failures = []
        for result in await asyncio.gather(*(run_task(task) for task in tasks)):
            failures.extend(result)

        return failures

if __name__ == "__main__":
    formatter = PythonFormatter(sys.argv[1:])
    sys.exit(formatter.run())
