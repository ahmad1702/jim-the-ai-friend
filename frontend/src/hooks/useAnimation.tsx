import autoAnimate from '@formkit/auto-animate'
import { useEffect, useRef } from 'react'

function useAnimation<T extends HTMLElement>() {
    const ref = useRef<T>(null)
    useEffect(() => { ref.current && autoAnimate(ref.current) }, [ref])
    return ref;
}

export default useAnimation