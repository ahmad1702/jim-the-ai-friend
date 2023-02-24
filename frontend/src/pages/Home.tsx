import { Box } from '@chakra-ui/react'
import React from 'react'
import Hero from '../components/home/Hero'
import NavBar from '../components/Navbar'

type Props = {}

const Home = (props: Props) => {
  return (
    <Box>
        <NavBar />
        <Hero />
    </Box>
  )
}

export default Home