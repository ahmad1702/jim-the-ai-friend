import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { ChakraProvider, extendTheme, ThemeConfig } from '@chakra-ui/react'


// 2. Add your color mode config
const config: ThemeConfig = {
  initialColorMode: 'system',
  useSystemColorMode: false,
}
const colors = {
  brand: {
    50: 'rgb(250 245 255)',
    100: 'rgb(243 232 255)',
    200: 'rgb(233 213 255)',
    300: 'rgb(216 180 254)',
    400: 'rgb(192 132 252)',
    500: 'rgb(168 85 247)',
    600: 'rgb(147 51 234)',
    700: 'rgb(126 34 206)',
    800: 'rgb(107 33 168)',
    900: 'rgb(88 28 135)',
  }
}
const theme = extendTheme({ colors, config })
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ChakraProvider theme={theme}>
      <App />
    </ChakraProvider>
  </React.StrictMode>,
)
