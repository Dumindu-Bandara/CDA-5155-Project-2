from enum import Enum
import argparse

MEMORY_START = 256

PRE_ISSUE_BUFFER_SIZE = 4
PRE_ALU1_BUFFER_SIZE = 2
PRE_ALU2_BUFFER_SIZE = 1
PRE_ALU3_BUFFER_SIZE = 1
PRE_MEMORY_BUFFER_SIZE = 1
POST_ALU2_BUFFER_SIZE = 1
POST_ALU3_BUFFER_SIZE = 1
POST_MEMORY_BUFFER_SIZE = 1

MAX_FETCHES_PER_CYCLE = 2
MAX_ISSUES_PER_CYCLE = 3
MAX_ALU1_ISSUES_PER_CYCLE = 1
MAX_ALU2_ISSUES_PER_CYCLE = 1
MAX_ALU3_ISSUES_PER_CYCLE = 1


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

        # with open("disassembly.txt", "w") as outfile:
        #     for line in disassebly_output:
        #         outfile.write(line + "\n")


class ProcessorPipeline():

    def __init__(self):
        self.registers = [0] * 32
        self.memory = {}
        self.pc = MEMORY_START
        self.cycle = 1

        # Buffers
        self.pre_issue_prev = []
        self.pre_issue_next = []

        self.alu1_prev = []
        self.alu1_next = []

        self.memory_prev = []
        self.memory_next = []

        self.alu2_prev = []
        self.alu2_next = []

        self.post_alu2_prev = []
        self.post_alu2_next = []

        self.alu3_prev = []
        self.alu3_next = []

        self.post_alu3_prev = []
        self.post_alu3_next = []



        self.post_memory_prev = []
        self.post_memory_next = []

        # Stall Status
        self.fetch_stall_prev = False
        self.fetch_stall_curr = False

        # Pipeline status
        self.ended = False

    
    def instruction_fetch(self):


        if self.fetch_stall_prev:
            pass

        else:

            num_issues = min(PRE_ISSUE_BUFFER_SIZE - len(self.pre_issue_prev), MAX_FETCHES_PER_CYCLE)

            for i in range(num_issues):
                instruction = self.instructions.get(self.pc)
                decoded_instruction = instruction_decoder(instruction, self.pc)

                if decoded_instruction["operation"] == Category4Opcode.BREAK:
                    self.ended = True
                    break

                elif decoded_instruction["operation"] in [Category1Opcode.BEQ, Category1Opcode.BNE, Category1Opcode.BLT]:
                    # Branch instructions are not issued
                    # TODO: Add branch handling
                    break

                else:
                    self.pre_issue_next.append(decoded_instruction)

                self.pc += 4

    def instruction_issue(self):

        issue_count = 0
        alu1_issue_count = 0
        alu2_issue_count = 0
        alu3_issue_count = 0

        pop_index = 0



        while True:
            if issue_count >= MAX_ISSUES_PER_CYCLE:
                break

            instruction = self.pre_issue_prev[pop_index]

            # Hazard detection and issue logic
            # No structural hazards - no speculation - HZ-1
            # No RAW and WAW hazards with active instructions - HZ-2
            # No WAW or WAR hazards with two instructions in the same cycle - HZ-3
            # No WAR hazards with not-issued intructions - HZ-4
            # For MEM all source register are ready - HZ-5
            # Load stay until all previous stores? - HZ-6
            # In order store issue - HZ-7

            if instruction["operation"] in [Category3Opcode.LW, Category1Opcode.SW]:
                if alu1_issue_count < MAX_ALU1_ISSUES_PER_CYCLE and len(self.alu1_prev) < PRE_ALU1_BUFFER_SIZE:
                    # TODO: Add hazard detection here
                    # HZ-1, HZ-2, HZ-3, HZ-4, HZ-5, HZ-6, HZ-7
                    self.alu1_next.append(instruction)
                    alu1_issue_count += 1
                    issue_count += 1

            if instruction["operation"] in [Category2Opcode.ADD, Category2Opcode.SUB, Category3Opcode.ADDI]:
                if alu2_issue_count < MAX_ALU2_ISSUES_PER_CYCLE and len(self.alu2_next) < PRE_ALU2_BUFFER_SIZE:
                    # TODO: Add hazard detection here
                    # HZ-1, HZ-2, HZ-3, HZ-4
                    self.alu2_next.append(instruction)
                    alu2_issue_count += 1
                    issue_count += 1

            if instruction["operation"] in [Category2Opcode.AND, Category2Opcode.OR, Category3Opcode.ANDI, Category3Opcode.ORI, Category3Opcode.SLLI, Category3Opcode.SRAI]:
                if alu3_issue_count < MAX_ALU3_ISSUES_PER_CYCLE and len(self.alu3_next) < PRE_ALU3_BUFFER_SIZE:
                    # TODO: Add hazard detection here
                    # HZ-1, HZ-2, HZ-3, HZ-4
                    self.alu3_next.append(instruction)
                    alu3_issue_count += 1
                    issue_count += 1


    def alu1_execute(self):
        """Supports Category1Opcode.SW and Category3Opcode.LW instructions."""

        instruction = self.alu1_prev.pop(0)

        if instruction["operation"] == Category3Opcode.LW:
            memory_address = self.registers[instruction["rs1"]] + instruction["immediate"]
            instruction["memory_address"] = memory_address

        elif instruction["operation"] == Category1Opcode.SW:
            memory_address = self.registers[instruction["rs2"]] + instruction["immediate"]
            instruction["memory_address"] = memory_address

        self.memory_next.append(instruction)
    

    def alu2_execute(self):
        """Supports Category2Opcode.ADD, Category2Opcode.SUB, and Category3Opcode.ADDI instructions."""
        
        instruction = self.alu2_prev.pop(0)

        if instruction["operation"] == Category2Opcode.ADD:
            result = self.registers[instruction["rs1"]] + self.registers[instruction["rs2"]]
            instruction["result"] = result

        elif instruction["operation"] == Category2Opcode.SUB:
            result = self.registers[instruction["rs1"]] - self.registers[instruction["rs2"]]
            instruction["result"] = result
            
        elif instruction["operation"] == Category3Opcode.ADDI:
            result = self.registers[instruction["rs1"]] + instruction["immediate"]
            instruction["result"] = result

        self.post_alu2_next.append(instruction)

    def alu3_execute(self):
        """Supports Category2Opcode.AND, Category2Opcode.OR, Category3Opcode.ANDI, Category3Opcode.ORI, Category3Opcode.SLLI, and Category3Opcode.SRAI instructions."""
        
        instruction = self.alu3_prev.pop(0)

        if instruction["operation"] == Category2Opcode.AND:
            result = self.registers[instruction["rs1"]] & self.registers[instruction["rs2"]]
            instruction["result"] = result

        elif instruction["operation"] == Category2Opcode.OR:
            result = self.registers[instruction["rs1"]] | self.registers[instruction["rs2"]]
            instruction["result"] = result
            
        elif instruction["operation"] == Category3Opcode.ANDI:
            result = self.registers[instruction["rs1"]] & instruction["immediate"]
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.ORI:
            result = self.registers[instruction["rs1"]] | instruction["immediate"]
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.SLLI:
            result = self.registers[instruction["rs1"]] << instruction["immediate"]
            instruction["result"] = result

        elif instruction["operation"] == Category3Opcode.SRAI:
            result = self.registers[instruction["rs1"]] >> instruction["immediate"]
            instruction["result"] = result

        self.post_alu3_next.append(instruction)

    def memory_access(self):
        
        instruction = self.memory_prev.pop(0)
        if instruction["operation"] == Category3Opcode.LW:
            instruction["loaded_value"] = self.memory.get(instruction["memory_address"], 0)
        elif instruction["operation"] == Category1Opcode.SW:
            self.memory[instruction["memory_address"]] = self.registers[instruction["rs2"]]

    def write_back(self):
        
        post_mem_instruction = self.post_memory_prev.pop(0)

        if post_mem_instruction["operation"] == Category3Opcode.LW:
            self.registers[post_mem_instruction["rd"]] = post_mem_instruction["loaded_value"]

        else:
            post_alu2_instruction = self.post_alu2_prev.pop(0)
            self.registers[post_alu2_instruction["rd"]] = post_alu2_instruction["result"]

            post_alu3_instruction = self.post_alu3_prev.pop(0)
            self.registers[post_alu3_instruction["rd"]] = post_alu3_instruction["result"]

            
    def tick(self):
        """Advance the pipeline by one cycle."""
        
        # Update previous cycle buffers with next cycle buffers
        self.pre_issue_prev.extend(self.pre_issue_next)
        self.pre_issue_next = []

        self.alu1_prev.extend(self.alu1_next)
        self.alu1_next = []

        self.memory_prev = self.memory_next
        self.memory_next = []

        self.alu2_prev = self.alu2_next
        self.alu2_next = []

        self.post_alu2_prev = self.post_alu2_next
        self.post_alu2_next = []

        self.alu3_prev = self.alu3_next
        self.alu3_next = []

        self.post_alu3_prev = self.post_alu3_next
        self.post_alu3_next = []

        self.post_memory_prev = self.post_memory_next
        self.post_memory_next = []

        self.fetch_stall_prev = self.fetch_stall_curr
        self.fetch_stall_curr = False

        self.cycle += 1


    def process(self, riscv_text: str):
        with open(riscv_text, "r") as file:
            instructions = file.readlines()

            self.instructions = {
                MEMORY_START + i * 4: inst.strip()
                for i, inst in enumerate(instructions)
            }

        with open("simulation.txt", "w") as simfile:
            while True:
                self.instruction_fetch()

                if self.ended:
                    break

                self.instruction_issue()
                self.alu1_execute()
                self.alu2_execute()
                self.alu3_execute()
                self.memory_access()
                self.write_back()



                # instruction = instructions.get(self.PC)
                # decoded_instruction = instruction_decoder(instruction, self.PC)

                # self.execute_instruction(decoded_instruction)

                # # output_state = self.output_state(decoded_instruction)

                # if self.cycle != 1:
                #     simfile.write("\n")

                # simfile.write(output_state)

                # if decoded_instruction["operation"] == Category4Opcode.BREAK:
                #     break

                # self.cycle += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('risv_text', type=str, help='Path to the RISC-V assembly text file')
    args = parser.parse_args()
    riscv_instructions = args.risv_text

    disassembler = Disassembler()
    disassembler.disassemble(riscv_instructions)

    memory = disassembler.memory
    print("Hi")

    



if __name__ == "__main__":
    main()

