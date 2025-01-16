# test_inference.py
import subprocess
import sys
import os

def run_test(command):
    print(f"\033[94mRunning command: {' '.join(command)}\033[0m\n\n")
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        print("\033[92mTest passed.\033[0m")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("\033[91mTest failed.\033[0m")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)

def main():
    # Paths to dummy images (replace with actual test images paths)
    left_images = [
        'test_data/P0002_left_0.png',
        'test_data/P0002_left_1.png',
        'test_data/P0002_left_2.png',
        'test_data/P0002_left_3.png',
        'test_data/P0002_left_4.png',
        'test_data/P0002_left_5.png',
        'test_data/P0002_left_6.png',
        'test_data/P0002_left_7.png',
        'test_data/P0002_left_8.png',
        'test_data/P0002_left_9.png',
        'test_data/P0002_left_10.png',
        'test_data/P0002_left_11.png',
        'test_data/P0002_left_12.png',
        'test_data/P0002_left_13.png',
        'test_data/P0002_left_14.png',
        'test_data/P0002_left_15.png',
        'test_data/P0002_left_16.png',
        'test_data/P0002_left_17.png',
        'test_data/P0002_left_18.png',
        'test_data/P0002_left_19.png'
    ]
    right_images = [
        'test_data/P0002_right_0.png',
        'test_data/P0002_right_1.png',
        'test_data/P0002_right_2.png',
        'test_data/P0002_right_3.png',
        'test_data/P0002_right_4.png',
        'test_data/P0002_right_5.png',
        'test_data/P0002_right_6.png',
        'test_data/P0002_right_7.png',
        'test_data/P0002_right_8.png',
        'test_data/P0002_right_9.png',
        'test_data/P0002_right_10.png',
        'test_data/P0002_right_11.png',
        'test_data/P0002_right_12.png',
        'test_data/P0002_right_13.png',
        'test_data/P0002_right_14.png',
        'test_data/P0002_right_15.png',
        'test_data/P0002_right_16.png',
        'test_data/P0002_right_17.png',
        'test_data/P0002_right_18.png',
        'test_data/P0002_right_19.png'
    ]
    
    # Path to model weights 
    model_path = './checkpoints/HRTFNet.pth'
    # Output path
    output_path = 'test_output.sofa'

    # Test cases
    test_cases = [
        # {
        #     'description': 'Test with 3 images (Task 1)',
        #     'left': left_images[:3],
        #     'right': right_images[:3],
        #     'expected_task': 1
        # },
        # {
        #     'description': 'Test with 7 images (Task 2)',
        #     'left': left_images[:7],
        #     'right': right_images[:7],
        #     'expected_task': 2
        # },
        {
            'description': 'Test with 19 images (Task 3)',
            'left': left_images[:19],
            'right': right_images[:19],
            'expected_task': 3
        },
        {
            'description': 'Test with 5 images (Non-standard number)',
            'left': left_images[:5],
            'right': right_images[:5],
            'expected_task': 'Non-standard'
        },
        {
            'description': 'Test with mismatched number of images (Should fail)',
            'left': left_images[:3],
            'right': right_images[:4]  # Intentional mismatch
        },
    ]

    import os
    print(f"\033[93m{os.getcwd()}\033[0m")
    for idx, test_case in enumerate(test_cases):
        print(f"\n\033[96m=== Test Case {idx + 1}: {test_case['description']} ===\033[0m")
        command = [sys.executable, './scripts/inference.py',
                   '-l'] + test_case['left'] + \
                  ['-r'] + test_case['right'] + \
                  ['-o', output_path,
                   '--model_path', model_path]
        run_test(command)
        break
    

if __name__ == "__main__":
    main()