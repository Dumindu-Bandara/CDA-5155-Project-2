from enum import Enum
import argparse

MEMORY_START = 256

class DissasemblyState(Enum):
    """State variable which denotes whether the disassembly is in instruction or data mode."""

    INSTRUCTION = 1
    DATA = 2

class InstructionCategory(Enum):
    """Enumeration for instruction categories."""

    CATEGORY_1 = "00"
    CATEGORY_2 = "01"
    CATEGORY_3 = "10"
    CATEGORY_4 = "11"


class Category1Opcode(Enum):
    """Enumeration for Category 1 opcodes."""

    BEQ = "00000"
    BNE = "00001"
    BLT = "00010"
    SW = "00011"


class Category2Opcode(Enum):
    """Enumeration for Category 2 opcodes."""

    ADD = "00000"
    SUB = "00001"
    AND = "00010"
    OR = "00011"


class Category3Opcode(Enum):
    """Enumeration for Category 3 opcodes."""

    ADDI = "00000"
    ANDI = "00001"
    ORI = "00010"
    SLLI = "00011"
    SRAI = "00100"
    LW = "00101"


class Category4Opcode(Enum):
    """Enumeration for Category 4 opcodes."""

    JAL = "00000"
    BREAK = "11111"




def twos_complement(bin_str: str) -> int:
    """Convert a binary string in two's complement format to its integer value.

    Args:
        bin_str (str): The binary string to convert.
    Returns:
        int: The integer value of the binary string.
    """

    value = -int(bin_str[0]) * 2 ** (len(bin_str) - 1) + int(bin_str[1:], 2)
    return value


def instruction_decoder(instruction: str, address: int) -> dict[str, int | str | Enum]:
    """Decode a RISC-V instruction into its components.
    Args:
        instruction (str): The binary string representation of the instruction.
        address (int): The memory address of the instruction.
    Returns:
        dict: A dictionary containing the decoded instruction components.
    """

    output_dict = {}

    if instruction[30:32] == InstructionCategory.CATEGORY_1.value:
        opcode = instruction[25:30]
        immediate = instruction[0:7] + instruction[20:25]

        output_dict["immediate"] = twos_complement(immediate)
        output_dict["rs1"] = int(instruction[12:17], 2)
        output_dict["rs2"] = int(instruction[7:12], 2)
        output_dict["func3"] = "000"

        output_dict["category"] = InstructionCategory.CATEGORY_1
        operation = Category1Opcode(opcode)
        output_dict["operation"] = operation

        if operation == Category1Opcode.SW:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rs1']}, {output_dict['immediate']}(x{output_dict['rs2']})"
            )
        else:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rs1']}, x{output_dict['rs2']}, #{output_dict['immediate']}"
            )

    elif instruction[30:32] == InstructionCategory.CATEGORY_2.value:
        opcode = instruction[25:30]
        output_dict["rd"] = int(instruction[20:25], 2)
        output_dict["rs1"] = int(instruction[12:17], 2)
        output_dict["rs2"] = int(instruction[7:12], 2)
        output_dict["func3"] = "000"
        output_dict["func7"] = "0000000"

        output_dict["category"] = InstructionCategory.CATEGORY_2
        operation = Category2Opcode(opcode)
        output_dict["operation"] = operation

        output_dict["assembly"] = (
            f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rd']}, x{output_dict['rs1']}, x{output_dict['rs2']}"
        )

    elif instruction[30:32] == InstructionCategory.CATEGORY_3.value:
        opcode = instruction[25:30]
        output_dict["rd"] = int(instruction[20:25], 2)
        output_dict["rs1"] = int(instruction[12:17], 2)

        if opcode in [Category3Opcode.SLLI.value, Category3Opcode.SRAI.value]:
            immediate = instruction[7:12]
            output_dict["immediate"] = int(immediate, 2)
        else:
            immediate = instruction[0:12]
            output_dict["immediate"] = twos_complement(immediate)

        output_dict["func3"] = "000"

        output_dict["category"] = InstructionCategory.CATEGORY_3
        operation = Category3Opcode(opcode)
        output_dict["operation"] = operation

        if operation == Category3Opcode.LW:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rd']}, {output_dict['immediate']}(x{output_dict['rs1']})"
            )
        else:
            output_dict["assembly"] = (
                f"{instruction}\t{address}\t{operation.name.lower()} x{output_dict['rd']}, x{output_dict['rs1']}, #{output_dict['immediate']}"
            )

    elif instruction[30:32] == InstructionCategory.CATEGORY_4.value:
        opcode = instruction[25:30]

        output_dict["category"] = InstructionCategory.CATEGORY_4
        operation = Category4Opcode(opcode)
        output_dict["operation"] = operation

        if opcode == Category4Opcode.BREAK.value:
            output_dict["assembly"] = f"{instruction}\t{address}\tbreak"
        elif opcode == Category4Opcode.JAL.value:
            rd = int(instruction[20:25], 2)
            output_dict["rd"] = rd

            immediate = instruction[0:20]
            output_dict["immediate"] = twos_complement(immediate)
            output_dict["assembly"] = (
                f"{instruction}\t{address}\tjal x{output_dict['rd']}, #{output_dict['immediate']}"
            )

    return output_dict

class Disassembler:
    """Class to handle disassembly of RISC-V instructions and data."""

    def __init__(self):
        self.memory = {}
        self.state = DissasemblyState.INSTRUCTION

    def disassemble(self, riscv_text: str):
        """Disassemble the RISC-V instructions and data from the input file.

        Args:
            riscv_text (str): Path to the input file containing RISC-V instructions.
        """

        disassebly_output = []

        with open(riscv_text, "r") as file:
            instructions = file.readlines()

            address = MEMORY_START
            for instruction in instructions:
                instruction = instruction.strip()

                if self.state == DissasemblyState.INSTRUCTION:
                    decoded_instruction = instruction_decoder(instruction, address)
                    disassebly_output.append(decoded_instruction["assembly"])

                    if decoded_instruction["operation"] == Category4Opcode.BREAK:
                        self.state = DissasemblyState.DATA

                elif self.state == DissasemblyState.DATA:
                    data_value = twos_complement(instruction)
                    self.memory[address] = data_value
                    disassebly_output.append(f"{instruction}\t{address} {data_value}")

                address += 4

        with open("disassembly.txt", "w") as outfile:
            for line in disassebly_output:
                outfile.write(line + "\n")

def ALU1():
    pass

def ALU2():
    pass

def ALU3():
    pass

def Processor():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('risv_text', type=str, help='Path to the RISC-V assembly text file')
    args = parser.parse_args()
    riscv_instructions = args.risv_text

    disassembler = Disassembler()
    disassembler.disassemble(riscv_instructions)

    



if __name__ == "__main__":
    pass

