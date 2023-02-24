import { Button, Icon, useColorMode } from "@chakra-ui/react"
import { MoonIcon, SunIcon } from "@heroicons/react/24/solid";

const DarkModeToggle = () => {
    const { colorMode, toggleColorMode } = useColorMode()
    return (
        <Button onClick={toggleColorMode}>
            {colorMode === 'light' ? <Icon as={MoonIcon} /> : <Icon as={SunIcon} />}
        </Button>
    )
}

export default DarkModeToggle;