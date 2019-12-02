import numpy as np

# TODO: Document

def bold_max(data):
    for i in range(data.shape[1]):
        col = data[:, i]
        col = col.astype("float")
        max_idx = col.argmax()
        data[max_idx, i] = "\\mathbf{" + data[max_idx, i] + "}"

def main():
    input_filename = "./utils/table_input.csv"
    output_filename = "./utils/table_output.csv"
    output_delimiter = "$ &\t$"
    output_newline = "$ \\\\\n"
    data = np.loadtxt(input_filename, dtype="str", delimiter="\t").astype("<U100")
    #data = np.loadtxt(input_filename, dtype="str", delimiter="%\t")
    #data = np.loadtxt(input_filename, dtype="float", delimiter=",")
    #data = np.transpose(data)
    #print(data.dtype)
    #bold_max(data)
    np.savetxt(output_filename, data, fmt="%s", delimiter=output_delimiter, newline=output_newline)
    #np.savetxt(output_filename, data, fmt="%s", delimiter="\t")
    print(data)


if __name__ == "__main__":
    main()

