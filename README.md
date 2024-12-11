# GradeGuru70B

GradeGuru70B is an advanced project leveraging state-of-the-art AI models to assist in understanding, generating, and analyzing academic content. This repository contains the tools, resources, and codebase required to facilitate interactive learning and intelligent grading using large language models (LLMs).

## Features

- **State-of-the-Art Model Integration**: Built on cutting-edge language models for superior performance.
- **Intelligent Grading**: Automates grading with human-like reasoning and analysis.
- **Content Generation**: Generates customized academic content such as assignments, quizzes, and solutions.
- **Interactive Learning**: Provides a platform for enhancing understanding through AI-driven explanations and support.
- **Scalable and Efficient**: Optimized for large-scale academic datasets and tasks.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: PyTorch, Transformers, Hugging Face
- **Models**: Large Language Models (70B parameter scale)
- **Platform**: Jupyter Notebook, Colab
- **Tools**: Weights & Biases (WandB) for experiment tracking

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/subh-775/GradeGuru70B.git
    ```
2. Navigate to the project directory:
    ```bash
    cd GradeGuru70B
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Model Training
1. Prepare your dataset in the `data/` directory.
2. Use the training script to fine-tune the model:
    ```bash
    python train.py --config config.yaml
    ```

### Grading System
1. Place your input files (e.g., essays, assignments) in the `input/` directory.
2. Run the grading script:
    ```bash
    python grade.py --input input/ --output results/
    ```

### Content Generation
1. Provide a prompt for academic content:
    ```bash
    python generate.py --prompt "Explain Newton's Laws of Motion"
    ```
2. Outputs will be saved in the `output/` directory.

## Dataset

The dataset used for this project includes a combination of:
- Publicly available academic datasets
- Custom-built datasets for grading and content generation tasks

## Contributing

We welcome contributions to improve GradeGuru70B. Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes and push to your forked repository.
4. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:
- **Subhansh Malviya**  
  Email: [subhansh4268@gmail.com](mailto:subhansh4268@gmail.com)  
  GitHub: [subh-775](https://github.com/subh-775)

---

Contribute to making learning smarter with **GradeGuru70B**!
