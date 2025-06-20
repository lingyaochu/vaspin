"""Module to parse VASP OUTCAR files."""

from abc import ABC, abstractmethod
from itertools import repeat
from typing import Any, Dict, List, TextIO, Type

from vaspin.utils.datatype import SymTensor


class ParserState:
    """Holds state potentially needed by multiple handlers."""

    def __init__(self) -> None:
        """Initialize the parser state."""
        self.num_atoms: int = 0
        self.is_ionic_step_finished: bool = False


class InfoHandler(ABC):
    """Abstract base class for OUTCAR information handlers."""

    KEY: str

    def __init_subclass__(cls):
        """Register subclass in the HANDLERS dictionary."""
        super().__init_subclass__()
        if "KEY" not in cls.__dict__ or not isinstance(cls.KEY, str):
            raise TypeError(
                f"{cls.__name__} must define a class variable 'KEY' of type str."
            )
        HANDLERS[cls.KEY] = cls

    @property
    @abstractmethod
    def HEADER(self) -> str:
        """Header string to identify relevant lines in the OUTCAR file."""
        pass

    def matches(self, line: str) -> bool:
        """Check if the handler should process this line (based on header)."""
        return self.HEADER in line.strip()

    @abstractmethod
    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the relevant information starting from the matched line.

        Reads subsequent lines from f_iter if necessary.
        Updates the data dictionary and potentially the state object.
        """
        pass

    def _log(self, data: Dict[str, Any], message: str):
        """Helper to add logs to a common log list in the data dict."""
        if "_parse_log" not in data:
            data["_parse_log"] = []
        data["_parse_log"].append(f"{self.__class__.__name__}: {message}")

    def _skip_lines(self, f_iter: TextIO, num_lines: int):
        """Skip a number of lines in the file iterator."""
        for _ in repeat(None, num_lines):
            next(f_iter)


HANDLERS: Dict[str, Type[InfoHandler]] = {}


class NElectronHandler(InfoHandler):
    """Handles the number of electron in the OUTCAR file."""

    KEY = "N electrons"

    @property
    def HEADER(self) -> str:
        """Set header string for number of electrons."""
        return "total number of electrons"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the number of electrons."""
        self._log(data, "Parsing number of electrons.")
        electrons = float(line.strip().split()[2])
        data["N electrons"] = electrons


class DTensorHandler(InfoHandler):
    """Handles the ZFS D tensor block."""

    KEY = "D tensor"

    @property
    def HEADER(self) -> str:
        """Set header string for D tensor block."""
        return "Spin-spin contribution to zero-field splitting tensor (MHz)"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the D tensor block."""
        self._log(data, "Parsing D tensor block.")
        self._skip_lines(f_iter, 4)

        dataline = next(f_iter).strip()
        parts_sym = [float(x) for x in dataline.split()]
        d_tensor = SymTensor().from_sequence(parts_sym)

        data["D tensor"] = d_tensor.get_matrix_sym().tolist()


class NIonsHandler(InfoHandler):
    """Handles the number of ions in the OUTCAR file."""

    KEY = "N ions"

    @property
    def HEADER(self) -> str:
        """Set header string for number of ions."""
        return "number of ions     NIONS"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the number of ions."""
        self._log(data, "Parsing number of ions.")
        ions = int(line.strip().split()[-1])
        data["N ions"] = ions
        state.num_atoms = ions


class ForceHandler(InfoHandler):
    """Handles the forces on atoms"""

    KEY = "Forces"

    @property
    def HEADER(self) -> str:
        """Set header string for forces."""
        return "TOTAL-FORCE (eV/Angst)"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the forces on atoms."""
        self._log(data, "Parsing forces on atoms.")
        forces = []
        next(f_iter)

        for _ in repeat(None, state.num_atoms):
            line_str = next(f_iter).strip().split()
            # print(line_str)
            force = [float(i) for i in line_str[3:6]]
            forces.append(force)

        if "forces" not in data:
            data["forces"] = []
        data["forces"].append(forces)


class IonicEnergyHandler(InfoHandler):
    """Handles the ionic energy in the OUTCAR file."""

    KEY = "Energy"

    @property
    def HEADER(self) -> str:
        """Set header string for ionic energy."""
        return "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the energy of each ionic step."""
        self._log(data, "Parsing ionic energy.")
        self._skip_lines(f_iter, 3)
        energy = float(next(f_iter).strip().split()[-1])
        if "Ionic energy" not in data:
            data["Ionic energy"] = []
        data["Ionic energy"].append(energy)


class PhononHandler(InfoHandler):
    """Handles the phonon frequencies and eigenvectors."""

    KEY = "Phonon"

    @property
    def HEADER(self) -> str:
        """Set header string for phonon frequencies."""
        return "Eigenvectors and eigenvalues of the dynamical matrix"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the phonon frequencies and eigenvectors."""
        self._log(data, "Parsing phonon frequencies.")

        data["phonon"] = {"frequencies": [], "eigenmodes": []}

        self._skip_lines(f_iter, 3)

        for _ in repeat(None, 3 * state.num_atoms):
            # the frequency in THz
            line_str = next(f_iter).strip().split()
            freq = float(line_str[3]) if line_str[1] == "f" else -float(line_str[2])
            data["phonon"]["frequencies"].append(freq)

            # the eigenvectors
            next(f_iter)
            mode = []
            for _ in repeat(None, state.num_atoms):
                line_str = next(f_iter).strip().split()
                mode.append([float(i) for i in line_str[3:6]])
            next(f_iter)
            data["phonon"]["eigenmodes"].append(mode)


class DielectricEleHandler(InfoHandler):
    """Handles the electron contribution to dielectric tensor."""

    KEY = "Dielectric ele"

    @property
    def HEADER(self) -> str:
        """Set header string for dielectric tensor."""
        return (
            "MACROSCOPIC STATIC DIELECTRIC TENSOR "
            "(including local field effects in DFT)"
        )

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the dielectric tensor."""
        # the electronic part appears twice, skip the second
        if "dielectric_ele" in data:
            self._log(data, "The electronic part has been parsed, skip.")
            return

        self._log(data, "Parsing dielectric tensor, the electronic part.")
        self._skip_lines(f_iter, 1)

        for _i in repeat(None, 3):
            line_str = next(f_iter).strip().split()
            row = [float(x) for x in line_str]
            if "dielectric_ele" not in data:
                data["dielectric_ele"] = []
            data["dielectric_ele"].append(row)


class DielectricIonHandler(InfoHandler):
    """Handles the ionic contribution to dielectric tensor."""

    KEY = "Dielectric ion"

    @property
    def HEADER(self) -> str:
        """Set header string for dielectric tensor."""
        return "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the ionic part of dielectric tensor."""
        self._log(data, "Parsing dielectric tensor, the ionic part.")
        self._skip_lines(f_iter, 1)

        for _i in repeat(None, 3):
            line_str = next(f_iter).strip().split()
            row = [float(x) for x in line_str]
            if "dielectric_ion" not in data:
                data["dielectric_ion"] = []
            data["dielectric_ion"].append(row)


class HyperfineFermiHandler(InfoHandler):
    """Handles the hyperfine interaction parameters from fermi contact."""

    KEY = "Hyperfine fermi"

    @property
    def HEADER(self) -> str:
        """Set header string for hyperfine parameters."""
        return "Fermi contact (isotropic) hyperfine coupling parameter (MHz)"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the hyperfine interaction parameters from fermi contact."""
        self._log(data, "Parsing hyperfine interaction parameters from fermi contact.")
        self._skip_lines(f_iter, 3)

        A_pw = []
        A_ps = []
        A_ae = []
        A_c = []
        for _ in repeat(None, state.num_atoms):
            line_str = next(f_iter).strip().split()
            A_pw.append(float(line_str[1]))
            A_ps.append(float(line_str[2]))
            A_ae.append(float(line_str[3]))
            A_c.append(float(line_str[4]))

        data["hyperfine_fermi"] = {"A_pw": A_pw, "A_ps": A_ps, "A_ae": A_ae, "A_c": A_c}


class HyperfineDipolarHandler(InfoHandler):
    """Handles the hyperfine interaction parameters from dipolar interaction."""

    KEY = "Hyperfine dipolar"

    @property
    def HEADER(self) -> str:
        """Set header string for hyperfine parameters."""
        return "Dipolar hyperfine coupling parameters (MHz)"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ) -> None:
        """Parse the hyperfine interaction parameters from dipolar interaction."""
        self._log(
            data, "Parsing hyperfine interaction parameters from dipolar interaction."
        )
        self._skip_lines(f_iter, 3)

        hyper_dipolar = []
        for _ in repeat(None, state.num_atoms):
            line_str = next(f_iter).strip().split()
            parts_sym = [float(x) for x in line_str[1:7]]
            dioplar_tensor = SymTensor().from_sequence(parts_sym)

            hyper_dipolar.append(dioplar_tensor.get_matrix_sym().tolist())

        data["hyperfine_dipolar"] = hyper_dipolar


class SitePotHandler(InfoHandler):
    """Handles the site potentials for each atoms in the last ionic step"""

    KEY = "Site potential"

    @property
    def HEADER(self) -> str:
        """Set header string for site potentials."""
        return "the norm of the test charge is"

    def parse(
        self, line: str, f_iter: TextIO, data: Dict[str, Any], state: ParserState
    ):
        """Parse the site potentials for each atom in the last ionic step."""
        self._log(data, "Parsing site potentials for each atom.")

        site_pot = []
        while True:
            line_str = next(f_iter).strip()
            if line_str == "":
                break

            parts = line_str.split()
            values = [float(parts[i]) for i in range(1, len(parts), 2)]
            site_pot.extend(values)

        data["site_potential"] = site_pot


class VaspOutcarParser:
    """Parses VASP OUTCAR files using a modular handler system."""

    def __init__(self, outcar_path: str):
        """Initialize the parser with the path to the OUTCAR file."""
        if not isinstance(outcar_path, str):
            raise TypeError("The OUTCAR path must be a string.")

        self.outcar_path: str = outcar_path
        self.data: Dict[str, Any] = {}
        self._state: ParserState = ParserState()
        # Register handler instances
        self._handlers: List[InfoHandler] = self._initialize_handlers()

    def _initialize_handlers(self) -> List[InfoHandler]:
        """Creates and returns a list of handler instances."""
        # The order might matter if some headers are substrings of others,
        # or if some handlers rely on state set by previous ones (like NIONS).
        # Place more specific or prerequisite handlers first.
        return [
            NIonsHandler(),
            NElectronHandler(),
            IonicEnergyHandler(),
            ForceHandler(),
            PhononHandler(),
            DTensorHandler(),
        ]

    def set_handlers(self, handler_keys: List[str]):
        """Set custom handlers to be used for parsing."""
        if len(handler_keys) == 0:
            raise ValueError("At least one handler key must be provided.")
        handlers = []

        for handler_key in handler_keys:
            if handler_key not in HANDLERS:
                raise ValueError(
                    f"Handler '{handler_key}' is not registered."
                    f"Please choose from: {list(HANDLERS.keys())}"
                )
            handlers.append(HANDLERS[handler_key]())

        self._handlers = handlers

    def _process_line(self, line: str, f: TextIO):
        """Process a single line from the OUTCAR file."""
        if not line.strip():
            return
        first_match = next(
            (handler for handler in self._handlers if handler.matches(line)), None
        )
        if first_match:
            first_match.parse(line, f, self.data, self._state)

    def parse(self, verbose: bool = False):
        """Parses the OUTCAR file using the registered handlers."""
        self.data = {"_parse_log": []}  # Reset data, initialize log
        self._state = ParserState()  # Reset state

        self._log(f"Starting parsing of '{self.outcar_path}'")
        with open(self.outcar_path, "r") as f:
            while True:
                try:
                    line = next(f)
                    self._process_line(line, f)
                except StopIteration:
                    break

        self._log("Finished parsing file.")
        if verbose:
            for msg in self.log:
                print(msg)

    def _log(self, message: str):
        """Adds a message to the internal parse log."""
        self.data["_parse_log"].append(f"Parser: {message}")

    @property
    def dmat(self) -> List[List[float]]:
        """Returns the D matrix from the parsed data."""
        return self.data.get("D tensor", [])

    @property
    def force(self) -> List[List[List[float]]]:
        """Returns the forces from the parsed data."""
        return self.data.get("forces", [])

    @property
    def energy(self) -> List[float]:
        """Returns the ionic energy from the parsed data."""
        return self.data.get("Ionic energy", [])

    @property
    def log(self) -> List[str]:
        """Returns the parse log."""
        return self.data.get("_parse_log", [])
